# Descriptive notes for each new metadata version. These are stored in a Python dict
# rather than as a comment, so that the existence of a description can be enforced by
# the unit tests when new versions are added.
VERSION_NOTES = {
    23: 'DataFrame.from_clause',
    22: 'TableMd/ColumnMd.media_validation',
    21: 'Separate InlineArray and InlineList',
    20: 'Store DB timestamps in UTC',
    19: 'UDF renames; ImageMemberAccess removal',
    18: 'Restructured index metadata',
    17: 'Renamed remotes to external_stores',
    16: 'Query functions; deferred Expr deserialization',
    15: 'Remotes in table metadata',
}
