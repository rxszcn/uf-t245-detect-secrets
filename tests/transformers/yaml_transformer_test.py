import textwrap
from unittest import mock

import pytest

from detect_secrets.core import scan
from detect_secrets.settings import transient_settings
from detect_secrets.transformers.exceptions import ParsingError
from detect_secrets.transformers.yaml import YAMLFileParser
from detect_secrets.transformers.yaml import YAMLTransformer
from testing.mocks import mock_file_object
from testing.mocks import mock_named_temporary_file


class TestYAMLTransformer:
    @staticmethod
    def test_basic():
        file = mock_file_object(
            textwrap.dedent("""
                keyA: string
                keyB: string    # with comments

                keyC: !!binary YWJjZGVm
                keyD: !!binary YWJjZGVm     # with comments
                keyE: !!binary invalidBinar

                dict:
                    keyD: nested string

                num: 1  # don't care
            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            'keyA: "string"',
            'keyB: "string"    # with comments',
            '',
            'keyC: "YWJjZGVm"',
            'keyD: "YWJjZGVm"     # with comments',
            'keyE: "invalidBinar"',
            '',
            '',
            'keyD: "nested string"',
        ]

    @staticmethod
    @pytest.mark.xfail(reason='TODO')
    @pytest.mark.parametrize(
        'block_chomping',
        ('', '-', '+'),
    )
    def test_multiline_block_scalar_folded_style(block_chomping):
        # NOTE(2020-11-07|domanchi): For YAML parsing, we don't really care about "literal" style
        # (the one with `|`) since that will keep new lines, and our assumption is that secrets
        # won't have new lines.
        #
        # However, "folded" style may be used to keep a certain line limit with very long secrets,
        # so we should probably handle that.
        file = mock_file_object(
            textwrap.dedent(
                f"""
                multiline: |{block_chomping}    # example
                    this is
                    a basic multiline string
            """,
            )[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            'multiline: this is a basic multiline string    # example',
        ]

    @staticmethod
    @pytest.mark.xfail(reason='TODO')
    @pytest.mark.parametrize(
        'block_chomping',
        ('', '-', '+'),
    )
    def test_multiline_block_scalar_literal_style(block_chomping):
        file = mock_file_object(
            textwrap.dedent("""
                multiline: >
                    this will be skipped
            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == ['']

    @staticmethod
    def test_single_line_flow_mapping():
        file = mock_file_object(
            textwrap.dedent("""
            object:
                keyA: 1
                dictionary:
                    - {keyB: valueB, keyC: valueC, keyD: valueD}
            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            '',
            '',
            '',
            'keyB: "valueB"',
            'keyC: "valueC"',
            'keyD: "valueD"',
        ]

    @staticmethod
    def test_multi_line_flow_mapping():
        file = mock_file_object(
            textwrap.dedent("""
            object:
                keyA: 1
                dictionary:
                    - {keyB: valueB, keyC: valueC, keyD: valueD}

            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            '',
            '',
            '',
            'keyB: "valueB"',
            'keyC: "valueC"',
            'keyD: "valueD"',
        ]

    @staticmethod
    def test_single_anchor_tag():
        file = mock_file_object(
            textwrap.dedent("""
                keyA: &test
                    keyB: string    # with comments
            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            '',
            'keyB: "string"    # with comments',
        ]

    @staticmethod
    def test_anchor_tag_alias_combination():
        file = mock_file_object(
            textwrap.dedent("""
                groupA: &groupA
                    keyA: valueA
                    keyB: valueB

                groupB: &groupB
                    keyC: valueC
                    keyD: *groupA
            """)[1:-1],
        )

        assert YAMLTransformer().parse_file(file) == [
            '',
            'keyA: "valueA"',
            'keyB: "valueB"',
            '',
            '',
            'keyC: "valueC"',
        ]


class TestYAMLTransformerMultiDocument:
    """
    Locked down behavior for `---`-separated YAML streams.

    Both documents contain an unquoted high-entropy value that only the transformer can
    surface (the entropy plugins require quotes). Line numbers asserted here are relative to
    the original file, which is what auditing relies on.
    """

    MULTI_DOCUMENT = textwrap.dedent("""\
        apiVersion: v1
        kind: Secret
        metadata:
            name: first
        data:
            token: Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A
        ---
        apiVersion: v1
        kind: Secret
        metadata:
            name: second
        data:
            token: pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS
    """)

    @staticmethod
    def test_both_documents_are_transformed():
        lines = YAMLTransformer().parse_file(mock_file_object(
            TestYAMLTransformerMultiDocument.MULTI_DOCUMENT,
        ))

        assert lines == [
            'apiVersion: "v1"',
            'kind: "Secret"',
            '',
            'name: "first"',
            '',
            'token: "Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A"',
            '',
            'apiVersion: "v1"',
            'kind: "Secret"',
            '',
            'name: "second"',
            '',
            'token: "pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS"',
        ]

    @staticmethod
    def test_line_numbers_match_original_file():
        lines = YAMLTransformer().parse_file(mock_file_object(
            TestYAMLTransformerMultiDocument.MULTI_DOCUMENT,
        ))

        # 1-indexed positions of the two secrets in the original file.
        assert lines[5] == 'token: "Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A"'
        assert lines[12] == 'token: "pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS"'

        # The output has the same number of lines as the source file, so scanning cannot
        # drift onto the wrong line.
        assert len(lines) == len(
            TestYAMLTransformerMultiDocument.MULTI_DOCUMENT.splitlines(),
        )

    @staticmethod
    def test_empty_documents_are_tolerated():
        content = textwrap.dedent("""\
            ---
            a: one
            ---
            ---
            b: two
        """)

        assert YAMLTransformer().parse_file(mock_file_object(content)) == [
            '',
            'a: "one"',
            '',
            '',
            'b: "two"',
        ]

    @staticmethod
    def test_directive_scoped_to_following_document():
        content = textwrap.dedent("""\
            %YAML 1.1
            ---
            a: one
            ---
            b: two
        """)

        lines = YAMLTransformer().parse_file(mock_file_object(content))

        # The directive and the marker occupy lines 1-2; `a` stays on line 3. The second
        # document starts on line 4.
        assert lines == [
            '',
            '',
            'a: "one"',
            '',
            'b: "two"',
        ]

    @staticmethod
    def test_content_before_first_marker_is_its_own_document():
        content = textwrap.dedent("""\
            a: one
            ---
            b: two
        """)

        assert YAMLTransformer().parse_file(mock_file_object(content)) == [
            'a: "one"',
            '',
            'b: "two"',
        ]


class TestYAMLTransformerMalformedMultiDocument:
    """
    A single malformed document must not invalidate the whole file.

    Semantics: malformed documents are emitted as raw source lines (so line-based plugins
    can still scan them), while well-formed documents keep full transformer semantics. All
    output is pinned to the original line numbers.
    """

    @staticmethod
    @pytest.mark.parametrize(
        'content, expected_lines',
        (
            # Malformed document first.
            (
                textwrap.dedent("""\
                    apiVersion: v1
                    kind: Broken
                    metadata:
                        name: {{ .values.name }}
                    ---
                    apiVersion: v1
                    kind: Secret
                    data:
                        token: pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS
                """),
                [
                    'apiVersion: v1',
                    'kind: Broken',
                    'metadata:',
                    '    name: {{ .values.name }}',
                    '',
                    'apiVersion: "v1"',
                    'kind: "Secret"',
                    '',
                    'token: "pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS"',
                ],
            ),
            # Malformed document second.
            (
                textwrap.dedent("""\
                    apiVersion: v1
                    kind: Secret
                    data:
                        token: Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A
                    ---
                    apiVersion: v1
                    kind: Broken
                    metadata:
                        name: {{ .values.name }}
                """),
                [
                    'apiVersion: "v1"',
                    'kind: "Secret"',
                    '',
                    'token: "Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A"',
                    '---',
                    'apiVersion: v1',
                    'kind: Broken',
                    'metadata:',
                    '    name: {{ .values.name }}',
                ],
            ),
        ),
    )
    def test_valid_document_still_transformed(content, expected_lines):
        assert YAMLTransformer().parse_file(mock_file_object(content)) == expected_lines

    @staticmethod
    def test_malformed_document_keeps_its_line_numbers():
        content = textwrap.dedent("""\
            metadata:
                name: {{ .values.name }}
            ---
            token: Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A
        """)

        lines = YAMLTransformer().parse_file(mock_file_object(content))

        assert len(lines) == len(content.splitlines())
        assert lines[0] == 'metadata:'
        assert lines[1] == '    name: {{ .values.name }}'
        assert lines[3] == 'token: "Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A"'

    @staticmethod
    def test_flow_mapping_in_valid_document_stays_on_one_line():
        # Rendering one document in isolation may not expand a single source line into
        # several, otherwise every following line number would shift.
        content = textwrap.dedent("""\
            a: {b: "1", c: "2"}
            ---
            metadata:
                name: {{ .values.name }}
        """)

        lines = YAMLTransformer().parse_file(mock_file_object(content))

        assert len(lines) == len(content.splitlines())
        assert lines[0] == 'b: "1" c: "2"'

    @staticmethod
    def test_directives_stay_with_their_document_in_degraded_mode():
        # The first document carries a directive; the second is malformed. The directive
        # must not be attributed to the previous document's chunk.
        content = textwrap.dedent("""\
            %YAML 1.1
            ---
            a: one
            ---
            b: {{ x }}
        """)

        lines = YAMLTransformer().parse_file(mock_file_object(content))

        assert len(lines) == len(content.splitlines())
        assert lines[2] == 'a: "one"'
        assert lines[4] == 'b: {{ x }}'

    @staticmethod
    def test_all_documents_malformed_raises_parsing_error():
        content = textwrap.dedent("""\
            a: {{ x }}
            ---
            b: {{ y }}
        """)

        with pytest.raises(ParsingError):
            YAMLTransformer().parse_file(mock_file_object(content))

    @staticmethod
    def test_single_malformed_document_still_raises_parsing_error():
        # Preserves the legacy all-or-nothing behavior for single-document files.
        with pytest.raises(ParsingError):
            YAMLTransformer().parse_file(
                mock_file_object('metadata:\n    name: {{ .values.name }}\n'),
            )


class TestYAMLTransformerSingleDocumentBaseline:
    """
    Guardrails for the pre-existing single-document behavior, including the eager fallback
    in the scanner's two-pass contract.
    """

    @staticmethod
    def test_broken_single_document_falls_back_to_raw_lines():
        content = 'metadata:\n    name: {{ .values.name }}\n'

        with mock_named_temporary_file(suffix='.yaml') as f:
            f.write(content.encode())
            f.seek(0)

            assert not list(scan.scan_file(f.name))

    @staticmethod
    def test_single_document_token_still_found():
        content = textwrap.dedent("""\
            kind: Secret
            data:
                token: Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A
        """)

        with transient_settings({
            'plugins_used': [{'name': 'Base64HighEntropyString', 'limit': 4.5}],
            'filters_used': [],
        }):
            with mock_named_temporary_file(suffix='.yaml') as f:
                f.write(content.encode())
                f.seek(0)

                assert [
                    (secret.line_number, secret.secret_value)
                    for secret in scan.scan_file(f.name)
                ] == [(3, 'Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A')]


class TestYAMLTransformerEndToEnd:
    """
    Verifies the two-pass contract in `scan.py` end to end: the transformer is used in the
    non-eager pass, and reported line numbers refer to the original file.
    """

    @staticmethod
    def _scan(content):
        with transient_settings({
            'plugins_used': [{'name': 'Base64HighEntropyString', 'limit': 4.5}],
            'filters_used': [],
        }):
            with mock_named_temporary_file(suffix='.yaml') as f:
                f.write(content.encode())
                f.seek(0)

                return {
                    secret.secret_value: secret.line_number
                    for secret in scan.scan_file(f.name)
                }

    @staticmethod
    def test_both_documents_are_reported_at_original_line_numbers():
        secrets = TestYAMLTransformerEndToEnd._scan(
            TestYAMLTransformerMultiDocument.MULTI_DOCUMENT,
        )

        assert secrets == {
            'Zx9Qm2Lp7Rt4Vw8YkN3Bd6Hf1Jc5Kg0A': 6,
            'pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS': 13,
        }

    @staticmethod
    def test_malformed_document_does_not_hide_valid_one():
        content = textwrap.dedent("""\
            apiVersion: v1
            kind: Broken
            metadata:
                name: {{ .values.name }}
            ---
            apiVersion: v1
            kind: Secret
            data:
                token: pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS
        """)

        assert TestYAMLTransformerEndToEnd._scan(content) == {
            'pQ7vN3xR9wT2yK8mB4cD6fG1hJ5kL0zS': 9,
        }


class TestYAMLFileParser:
    @staticmethod
    def test_basic():
        file = mock_file_object(
            textwrap.dedent("""
                keyA: string
                dict:
                    keyB: 123
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'keyA': {
                '__value__': 'string',
                '__line__': 1,
                '__original_key__': 'keyA',
            },

            # Ignores non-string or non-binary
            'dict': {
                'keyB': 123,
            },
        }

    @staticmethod
    @pytest.mark.parametrize(
        'block_scalar_style',
        ('>', '|'),
    )
    @pytest.mark.parametrize(
        'block_chomping',
        ('', '-', '+'),
    )
    def test_multi_line(block_scalar_style, block_chomping):
        # NOTE: Referenced https://yaml-multiline.info/ for the many ways to do multi line strings
        file = mock_file_object(
            textwrap.dedent(
                f"""
                key: {block_scalar_style}{block_chomping}   # comment
                    multi
                    # line
                    string
            """,
            )[1:-1],
        )

        assert [item.line for item in YAMLFileParser(file)] == [
            f'key: {block_scalar_style}{block_chomping}   # comment',
        ]

    @staticmethod
    @pytest.mark.parametrize(
        ['yaml_value', 'expected_value'],
        [
            ('string_value', 'string_value'),
            ('!!binary YWJjZGVm', b'abcdef'),
        ],
    )
    def test_possible_secret_format(yaml_value, expected_value):
        content = 'key: {yaml_value}'.format(yaml_value=yaml_value)
        f = mock_file_object(content)

        result = YAMLFileParser(f).json()
        assert result['key'] == {
            '__value__': expected_value,
            '__line__': mock.ANY,
            '__original_key__': mock.ANY,
        }

    @staticmethod
    @pytest.mark.parametrize(
        'content, expected',
        (
            # NOTE: The trailing new lines are important here!
            # It needs to be a string value, since we ignore non-string values (because we assume
            # secrets will be strings). However, the combination of the dictionary start character
            # `{` and the keys being on the same line causes unexpected results (see #374).
            (
                textwrap.dedent("""
                    { a: "1" }
                """)[1:],
                ['a: "1"'],
            ),
            (
                textwrap.dedent("""
                    a:
                        {b: "2"}
                """)[1:],
                ['', 'b: "2"'],
            ),
            (
                textwrap.dedent("""
                    a:
                    - {b: "2"}
                """)[1:],
                ['', 'b: "2"'],
            ),
            # New lines aren't important here, but since the first key is on the same line
            # as the start of the block, it will be handled funkily.
            (
                textwrap.dedent("""
                    {a: "1",
                        b: "2",
                    }
                """)[1:-1],
                ['a: "1"', 'b: "2"'],
            ),
            (
                textwrap.dedent("""
                    {
                        a: "1",
                        b: "2",
                    }
                """)[1:],
                ['', 'a: "1"', 'b: "2"'],
            ),
            (
                textwrap.dedent("""
                    { a: "1", b: "2" }
                """)[1:],
                ['a: "1"', 'b: "2"'],
            ),
        ),
    )
    def test_inline_dictionary(content, expected):
        lines = YAMLTransformer().parse_file(mock_file_object(content))
        assert lines == expected

    @staticmethod
    def test_single_line_flow_mapping():
        file = mock_file_object(
            textwrap.dedent("""
            dictionary:
                - {keyA: valueA, keyB: valueB, keyC: valueC}
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'dictionary': [
                {
                    'keyA': {
                        '__value__': 'valueA',
                        '__line__': 2,
                        '__original_key__': 'keyA',
                    },
                    'keyB': {
                        '__value__': 'valueB',
                        '__line__': 2,
                        '__original_key__': 'keyB',
                    },
                    'keyC': {
                        '__value__': 'valueC',
                        '__line__': 2,
                        '__original_key__': 'keyC',
                    },
                },
            ],
        }

    @staticmethod
    def test_multi_line_flow_mapping():
        file = mock_file_object(
            textwrap.dedent("""
            dictionary:
                - {keyA: valueA, keyB: valueB, keyC: valueC}

            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'dictionary': [
                {
                    'keyA': {
                        '__value__': 'valueA',
                        '__line__': 2,
                        '__original_key__': 'keyA',
                    },
                    'keyB': {
                        '__value__': 'valueB',
                        '__line__': 2,
                        '__original_key__': 'keyB',
                    },
                    'keyC': {
                        '__value__': 'valueC',
                        '__line__': 2,
                        '__original_key__': 'keyC',
                    },
                },
            ],
        }

    @staticmethod
    def test_inline_dictionary_same_starting_line():
        file = mock_file_object(
            textwrap.dedent("""
                {a: "1",
                    b: "2",
                }
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'a': {
                '__value__': '1',
                '__line__': 1,
                '__original_key__': 'a',
            },
            'b': {
                '__value__': '2',
                '__line__': 2,
                '__original_key__': 'b',
            },
        }

    @staticmethod
    def test_inline_dictionary_different_starting_line():
        file = mock_file_object(
            textwrap.dedent("""
                {
                    a: "1",
                    b: "2",
                }
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'a': {
                '__value__': '1',
                '__line__': 2,
                '__original_key__': 'a',
            },
            'b': {
                '__value__': '2',
                '__line__': 3,
                '__original_key__': 'b',
            },
        }

    @staticmethod
    def test_inline_empty_mapping_line_numbers():
        file = mock_file_object(
            textwrap.dedent("""
                a: {}
                b: "2"
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'a': {},
            'b': {
                '__value__': '2',
                '__line__': 2,
                '__original_key__': 'b',
            },
        }

    @staticmethod
    def test_inline_mapping_single_line_single_key_line_numbers():
        file = mock_file_object(
            textwrap.dedent("""
                a: {b: "2"}
                c: "3"
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'a': {
                'b': {
                    '__value__': '2',
                    '__line__': 1,
                    '__original_key__': 'b',
                },
            },
            'c': {
                '__value__': '3',
                '__line__': 2,
                '__original_key__': 'c',
            },
        }

    @staticmethod
    def test_inline_mapping_single_line_multikey_line_numbers():
        file = mock_file_object(
            textwrap.dedent("""
                a: {b: "2", c: "3"}
                d: "4"
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'a': {
                'b': {
                    '__value__': '2',
                    '__line__': 1,
                    '__original_key__': 'b',
                },
                'c': {
                    '__value__': '3',
                    '__line__': 1,
                    '__original_key__': 'c',
                },
            },
            'd': {
                '__value__': '4',
                '__line__': 2,
                '__original_key__': 'd',
            },
        }

    @staticmethod
    def test_single_anchor_tag():
        file = mock_file_object(
            textwrap.dedent("""
                keyA: &test
                    keyB: string    # with comments
                    keyC:
                        keyD: string
            """)[1:-1],
        )

        assert YAMLFileParser(file).json() == {
            'keyA': {
                'keyB': {
                    '__value__': 'string',
                    '__line__': 2,
                    '__original_key__': 'keyB',
                },
                'keyC': {
                    'keyD': {
                        '__value__': 'string',
                        '__line__': 4,
                        '__original_key__': 'keyD',
                    },
                },
            },
        }

    @staticmethod
    def test_anchor_tag_alias_combination():
        file = mock_file_object(
            textwrap.dedent("""
                groupA: &groupA
                    keyA: valueA
                    keyB: valueB

                groupB: &groupB
                    keyC: valueC
                    keyD: *groupA
            """)[1:-1],
        )

        temp = YAMLFileParser(file).json()
        assert temp == {
            'groupA': {
                'keyA': {
                    '__value__': 'valueA',
                    '__line__': 2,
                    '__original_key__': 'keyA',
                },
                'keyB': {
                    '__value__': 'valueB',
                    '__line__': 3,
                    '__original_key__': 'keyB',
                },
            },
            'groupB': {
                'keyC': {
                    '__value__': 'valueC',
                    '__line__': 6,
                    '__original_key__': 'keyC',
                },
                'keyD': {
                    'keyA': {
                        '__value__': 'valueA',
                        '__line__': 2,
                        '__original_key__': 'keyA',
                    },
                    'keyB': {
                        '__value__': 'valueB',
                        '__line__': 3,
                        '__original_key__': 'keyB',
                    },
                },
            },
        }
