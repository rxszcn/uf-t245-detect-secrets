import base64
import re
from collections import deque
from functools import lru_cache
from io import StringIO
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Pattern
from typing import Tuple
from typing import Union

import yaml
from yaml.tokens import FlowEntryToken
from yaml.tokens import KeyToken

from ..types import NamedIO
from ..util.filetype import determine_file_type
from ..util.filetype import FileType
from .base import BaseTransformer
from .exceptions import ParsingError


class YAMLTransformer(BaseTransformer):
    def should_parse_file(self, filename: str) -> bool:
        return determine_file_type(filename) == FileType.YAML

    def parse_file(self, file: NamedIO) -> List[str]:
        """
        :raises: ParsingError

        Multi-document YAML (e.g. `---`-separated manifests) is supported. Each document is
        parsed in isolation and pinned to the line span it occupies in the original stream,
        so line numbers always refer to the original file. Anchors and aliases are scoped
        per document by YAML itself, so per-document parsing does not lose anything there.
        Rendering keeps values that share a source line on a single output line, which is
        what makes the line numbers of later documents provably stable.

        Parsing per document also provides a natural degradation boundary: a malformed
        document falls back to raw line scanning on its own, while the rest of the file
        keeps full transformer semantics (entropy plugins, anchor/alias expansion, binary
        handling, comment preservation). Previously the whole file degraded to raw lines as
        soon as any document failed to parse.

        Files with fewer than two documents keep the legacy all-or-nothing behavior, so
        single-document rendering (including its line expansion for inline flow mappings)
        is unchanged.
        """
        content = file.read()
        chunks = _split_document_chunks(content)
        if len(chunks) <= 1:
            return _parse_single_document(content)

        return _parse_documents(chunks, _split_yaml_lines(content))


def _parse_single_document(content: str) -> List[str]:
    """
    :raises: ParsingError
    """
    try:
        items = sorted(
            YAMLFileParser(_string_io(content)),
            key=lambda x: x.line_number,
        )
    except yaml.YAMLError:
        raise ParsingError

    return _format_yaml_values(items)


def _parse_documents(
    chunks: List[Tuple[int, List[str]]],
    content_lines: List[str],
) -> List[str]:
    """
    Renders a multi-document stream document by document.

    Valid documents are rendered with the regular transformer semantics; malformed ones are
    emitted as raw source lines so that line-based plugins can still scan them at their
    original positions. Every document is pinned to the line span it occupies in the
    original file, which keeps reported line numbers correct.

    :raises: ParsingError: if no document could be parsed at all, preserving the scanner's
        existing fallback for completely broken files.
    """
    rendered_documents = [
        _render_document(chunk_lines, start_line, content_lines)
        for start_line, chunk_lines in chunks
    ]

    if all(rendered is None for rendered in rendered_documents):
        raise ParsingError

    lines: List[str] = []
    for (start_line, chunk_lines), rendered in zip(chunks, rendered_documents):
        # Chunks are contiguous, but a document may start after a preamble that produces no
        # document of its own (e.g. leading comments), so pad up to its original position.
        lines.extend([''] * (start_line - 1 - len(lines)))

        rendered_lines = chunk_lines if rendered is None else rendered
        lines.extend(rendered_lines)

        # Pad when the transformed document is shorter than its source (e.g. non-string
        # values are dropped), so that subsequent documents keep their line numbers.
        lines.extend([''] * (len(chunk_lines) - len(rendered_lines)))

    return lines


def _render_document(
    chunk_lines: List[str],
    start_line: int,
    content_lines: List[str],
) -> Optional[List[str]]:
    """
    Renders one document (as delimited by `---`) to scanner lines.

    :returns: None if the document is malformed. The result never spans more lines than the
        source document: values that share a source line are merged onto that one output
        line (see `_format_yaml_values`).
    """
    chunk_content = '\n'.join(chunk_lines)

    try:
        items = sorted(
            YAMLFileParser(
                _string_io(chunk_content),
                line_offset=start_line - 1,
                original_lines=content_lines,
            ),
            key=lambda x: x.line_number,
        )
    except yaml.YAMLError:
        return None

    rendered = _format_yaml_values(items, base_line=start_line, clamp_to_lines=len(chunk_lines))
    if len(rendered) > len(chunk_lines):     # pragma: no cover
        # Defensive only: clamping (above) and per-document parsing both guarantee that a
        # document never grows beyond its source span, so the line numbers of the documents
        # that follow it cannot shift.
        return None

    return rendered


def _format_yaml_values(
    items: List['YAMLValue'],
    base_line: int = 1,
    clamp_to_lines: Optional[int] = None,
) -> List[str]:
    """
    :param base_line: original-file line number that the *first* output line corresponds to.
        Line numbers in `items` are absolute within the original file, so padding is computed
        relative to this anchor. This lets the same routine render a whole file
        (`base_line=1`) or a single document placed further down the file.
    :param clamp_to_lines: if set, values that map to the same line are merged into a single
        output line, so the result never exceeds `clamp_to_lines` entries. This is required
        when rendering one document of a multi-document file in isolation, since expanding a
        line (e.g. a single-line flow mapping with several keys) would otherwise shift every
        line number that follows it.
    """
    seen = set()

    lines: List[str] = []
    for item in items:
        # Filter out previous lines seen before. This removes duplicates when it comes
        # to anchor & and alias * tags.
        if item in seen:
            continue
        else:
            seen.add(item)

        position = item.line_number - base_line
        rendered = _render_yaml_value(item)

        if (
            clamp_to_lines is not None
            and lines
            and position < len(lines)
        ):
            # Same source line as a previous value: keep them on one line so that the
            # document never grows beyond its original span.
            lines[position] = f'{lines[position]} {rendered}'
            continue

        while len(lines) < position:
            lines.append('')

        lines.append(rendered)

    return lines


def _render_yaml_value(item: 'YAMLValue') -> str:
    value = item.value
    if isinstance(value, bytes):
        # binary strings in YAML are base64 encoded. https://yaml.org/type/binary.html
        # While the YAML parser already decodes it for us, we want to capture the *raw*
        # base64 encoded value for two reasons:
        #   1. Increases coverage
        #      Our Base64HighEntropyString plugin is tuned for base64 strings. Including
        #      other potential characters excludes it from this scan, with no real
        #      potential gain. While the entropy limit may be different, the fact that
        #      the string is processable is a win already.
        #
        #   2. Supports audit functionality
        #      When we convert this value to its unicode representation, we need to
        #      performs several hacks in order to be able to find the raw binary string
        #      again, during our audit process. Keeping it to its original value simplifies
        #      this process.
        value = base64.b64encode(value).decode()

    line = item.line.strip()
    # TODO: parse the difference between block_scalar styles, and handle appropriately.
    # See test cases for more details.

    comment = ''
    matches = _yaml_comment_regex().search(line)
    if matches:
        comment = matches.group(0)

    # YAML does the parsing for us, and the value we'll receive is a string proper
    # (rather than the raw value, like `ConfigFileTransformer`). We want to artificially
    # add quotes here since we know they are strings, HighEntropyString will benefit
    # from this, and all other plugins don't care.
    #
    # However, if there is a quote inside, we need to escape it.
    value = value.replace('"', '\\"')

    return f'{item.key}: "{value}"{comment}'


def _split_document_chunks(content: str) -> List[Tuple[int, List[str]]]:
    """
    Splits a YAML stream into `(start_line, lines)` tuples, where `start_line` is 1-indexed
    and points at the first line of that document within the original file.

    Document boundaries are `---` markers at column 0 (the same rule PyYAML's scanner uses:
    the marker must be followed by whitespace or end-of-line). Directives that precede a
    marker (e.g. `%YAML`, `%TAG`) belong to the document that follows, so they are moved
    into the next chunk.
    """
    lines = _split_yaml_lines(content)
    marker_indices = [
        index
        for index, line in enumerate(lines)
        if _is_document_start_marker(line)
    ]

    if not marker_indices:
        return [(1, lines)] if lines else []

    starts: List[int] = []

    # Anything before the first `---` forms an implicit document, unless it only contains
    # directives, comments, or blank lines (which do not yield a document of their own).
    if any(not _is_ignorable_prelude(line) for line in lines[:marker_indices[0]]):
        starts.append(0)

    for marker_index in marker_indices:
        start = marker_index
        # Directives are only valid at the start of a document, so immediately preceding
        # `%...` lines are part of this document, not of the previous one. They can never
        # reach back past the preceding marker, since that marker's own chunk starts at or
        # before it.
        while start > 0 and _is_directive(lines[start - 1]):
            start -= 1

        starts.append(start)

    return [
        (
            start + 1,
            lines[start:starts[position + 1] if position + 1 < len(starts) else len(lines)],
        )
        for position, start in enumerate(starts)
    ]


def _is_document_start_marker(line: str) -> bool:
    return (
        line.startswith('---')
        and (len(line) == 3 or line[3] in ' \t\r\n\x85\u2028\u2029')
    )


def _is_directive(line: str) -> bool:
    return line.startswith('%')


def _is_ignorable_prelude(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith('#') or _is_directive(line)


@lru_cache(maxsize=1)
def _yaml_line_break_regex() -> Pattern:
    """
    PyYAML's Reader treats `\\r\\n`, `\\r`, `\\n`, `\\x85`, `\\u2028`, and `\\u2029` as line
    breaks. `str.splitlines()` also breaks on `\\v`, `\\f`, and `\\x1c`-`\\x1e`, which
    PyYAML does *not* count as new lines. Using the same rule as PyYAML keeps our
    line indexing (`item['__line__'] - 1`) aligned with the parser's marks.
    """
    return re.compile('\r\n|[\r\n\x85\u2028\u2029]')


def _split_yaml_lines(content: str) -> List[str]:
    lines = _yaml_line_break_regex().split(content)
    if lines and lines[-1] == '':
        # A trailing line break does not produce a final empty line.
        lines.pop()

    return lines


def _string_io(content: str) -> NamedIO:
    return cast(NamedIO, StringIO(content))


@lru_cache(maxsize=1)
def _yaml_comment_regex() -> Pattern:
    """
    From https://yaml-multiline.info/, it states that `#` cannot appear *after* a space
    or a newline, otherwise it will be a syntax error (for multiline strings that don't
    use a block scalar). This applies to single lines as well: for example, `a#b` will be
    treated as a single value, but `a #b` will only capture `a`, leaving `#b` as a comment.

    For lines that *do* use a block scalar, the YAML parser will throw a syntax error if
    there is additional text on the same line as the block scalar. Comments however, are fine.
    e.g.
        key: | # this is ok
        blah

        key: | but this is not
        blah

    Given that we've made it to this stage, we can assume the YAML file is syntactically
    correct. Therefore, if we add whitespace before the comment character, we can know that
    everything else *after* the comment character is a comment for a given line.
    """
    return re.compile(r'(\s+#[\S ]*)')


class YAMLValue(NamedTuple):
    key: str
    value: Union[str, bytes]
    line_number: int
    line: str


class YAMLFileParser:
    """
    Yaml config files are interesting, because they don't necessarily conform
    to our basic regex for detecting HighEntropyStrings as strings don't
    need to be quoted.

    This causes interesting issues, because our regex won't catch non-quoted
    strings, and if we ignore the quoting requirement, then we increase our
    false positive rate, because any long string would have high entropy.

    Therefore, we take a different approach: intercept the parsing of the yaml
    file to identify string values. This assumes:

        1. Secrets are strings or binaries
        2. Secrets are not keys

    Then, we calculate the entropy of those string values.

    The difficulty comes from determining the line number which these values
    come from. To do this, we transform the string into a dictionary of
    meta-tags, in the following format:

    >>> {
        'key': {
            '__value__': value,
            '__line__': <line_number>,
        }
    }

    This way, we can quickly identify the line number for auditing at a later
    stage.

    This parsing method is inspired by https://stackoverflow.com/a/13319530.
    """

    def __init__(
        self,
        file: NamedIO,
        line_offset: int = 0,
        original_lines: Optional[List[str]] = None,
    ):
        """
        :param line_offset: number of lines preceding `file` in the original document stream.
            Used when re-parsing a single document in isolation, so that reported line
            numbers remain relative to the original file.
        :param original_lines: lines of the original file, used to fetch inline comments.
            Defaults to the lines of `file` itself.
        """
        self.content = file.read()
        self.line_offset = line_offset
        self.original_lines = (
            original_lines
            if original_lines is not None
            else _split_yaml_lines(self.content)
        )

        self.loader = yaml.SafeLoader(self.content)
        self.loader.compose_node = self._compose_node_shim  # type: ignore

        self.is_inline_flow_mapping_key = False
        self.loader.parse_flow_mapping_key = self._parse_flow_mapping_key_shim  # type: ignore

    def json(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.loader.get_single_data())

    def _line_number_from_mark(self, line: int) -> int:
        """Converts a zero-indexed loader mark into a 1-indexed line in the original file."""
        return line + 1 + self.line_offset

    def __iter__(self) -> Iterator[YAMLValue]:
        """
        :returns: (value, line_number)

        All documents in the stream are composed, not just the first one, so that
        multi-document YAML (e.g. `---`-separated manifests) is scanned using the same
        semantics as a single document. Anchors and aliases are scoped per document by
        YAML, and PyYAML resets `anchors` after each document, so aliases keep working
        within each document.
        """
        # Used to obtain inline comments
        to_search: deque = deque()

        while self.loader.check_node():
            document = self.loader.get_data()
            if document is not None:
                to_search.append(document)

        while to_search:
            item: Any = to_search.popleft()

            if not item:
                # mainly for base case (e.g. if file is all comments)
                continue

            # If it doesn't have our meta-tags, it's not a value worth scanning.
            try:
                if '__line__' not in item:
                    if isinstance(item, str):
                        continue

                    # However, we need to recursively search in the dictionary for other such values
                    # that we may care about.
                    try:
                        to_search.extend(item.values())
                    except AttributeError:
                        # This is an array
                        to_search.extend(item)

                    continue
            except TypeError:
                # e.g. if item is a float.
                continue

            yield YAMLValue(
                key=item['__original_key__'],
                value=item['__value__'],
                line_number=item['__line__'],

                # We extract this separately because the parser drops the comments
                # (at least up to version 3.13).
                # https://github.com/yaml/pyyaml/blob/a2d481b8dbd2b352cb001f07091ccf669227290f/lib3/yaml/scanner.py#L749
                # The line value feeds into the filters, and helps us tune false positives.
                line=self.original_lines[item['__line__'] - 1],
            )

    def _compose_node_shim(
        self,
        parent: Optional[yaml.nodes.Node],
        index: Optional[yaml.nodes.Node],
    ) -> Optional[yaml.nodes.Node]:
        line = (
            self.loader.marks[-1].line
            if self.is_inline_flow_mapping_key
            else self.loader.line
        )

        node = yaml.composer.Composer.compose_node(self.loader, parent, index)  # type: ignore
        if node is None:
            return None

        node.__line__ = self._line_number_from_mark(line)    # type: ignore

        if node.tag.endswith(':map'):
            # Reset the inline flow mapping key when the end of a mapping is reached
            # to avoid complications with empty mappings
            self.is_inline_flow_mapping_key = False
            return _tag_dict_values(cast(yaml.nodes.MappingNode, node))

        # TODO: Not sure if need to do :seq

        return node

    def _parse_flow_mapping_key_shim(
        self,
        first: bool = False,
    ) -> yaml.nodes.Node:
        # There exists an edge case when a key and flow mapping start character `{` are on the same
        # line (Ex. '{key: value}) followed by an empty line. The parser will produce an off-by-one
        # error for the line number that it tracks internally. Since we track the start of the
        # mapping, we will use this line number when we are processing:
        # A) The first key in an inline dictionary where the flow mapping start character is on the
        # same line as the key
        # B) The n key of an inline dictionary that is followed by a FlowEntryToken (',') and
        # KeyToken ('key:')
        is_inline_dictionary = (
            first
            and self.loader.marks[-1].line == self.loader.peek_token().start_mark.line
            or self._check_next_tokens_shim(FlowEntryToken, KeyToken)
        )

        if is_inline_dictionary:
            self.is_inline_flow_mapping_key = True
        else:
            self.is_inline_flow_mapping_key = False

        return cast(yaml.nodes.Node, yaml.parser.Parser.parse_flow_mapping_key(self.loader, first))

    def _check_next_tokens_shim(
        self,
        *choices: Any,
    ) -> bool:
        """Check the next tokens type match the argument list of token types."""
        result = True
        i = 0

        if self.loader.tokens:
            if not choices:
                return result
            for choice in choices:
                if i < len(self.loader.tokens):
                    result = result and isinstance(self.loader.tokens[i], choice)
                    i += 1
        else:
            result = False

        return result


def _tag_dict_values(map_node: yaml.nodes.MappingNode) -> yaml.nodes.MappingNode:
    """
    :param map_node: It looks like map_node.value contains a list of
        pair tuples, corresponding to key,value pairs.
    """
    new_values = []
    for key, value in map_node.value:
        if not (
            value.tag.endswith(':str') or
            value.tag.endswith(':binary')
        ):
            new_values.append((key, value))
            continue

        augmented_string = yaml.nodes.MappingNode(
            tag=map_node.tag,
            value=[
                _create_key_value_pair_for_mapping_node_value(
                    key='__value__',
                    value=value.value,
                    tag=value.tag,
                ),
                _create_key_value_pair_for_mapping_node_value(
                    key='__line__',
                    value=str(value.__line__),
                    tag='tag:yaml.org,2002:int',
                ),
                _create_key_value_pair_for_mapping_node_value(
                    key='__original_key__',
                    value=key.value,
                    tag='tag:yaml.org,2002:str',
                ),
            ],
        )

        new_values.append((key, augmented_string))

    output = yaml.nodes.MappingNode(
        tag=map_node.tag,
        value=new_values,
        start_mark=map_node.start_mark,
        end_mark=map_node.end_mark,
        flow_style=map_node.flow_style,
    )
    return output


def _create_key_value_pair_for_mapping_node_value(
    key: str,
    value: Any,
    tag: str,
) -> Tuple[yaml.nodes.ScalarNode, yaml.nodes.ScalarNode]:
    return (
        yaml.nodes.ScalarNode(
            tag='tag:yaml.org,2002:str',
            value=key,
        ),
        yaml.nodes.ScalarNode(
            tag=tag,
            value=value,
        ),
    )
