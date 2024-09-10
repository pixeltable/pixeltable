# Descriptive notes for each new metadata version. These are stored in a Python dict
# rather than as a comment, so that the existence of a description can be enforced by
# the unit tests when new versions are added.
VERSION_NOTES = {
    20: 'Separate InlineArray and InlineList',
    19: 'UDF renames; ImageMemberAccess removal',
    18: 'Restructured index metadata',
    17: 'Renamed remotes to external_stores',
    16: 'Query functions; deferred Expr deserialization',
    15: 'Remotes in table metadata',
}
