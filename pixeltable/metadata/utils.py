from __future__ import annotations

from dataclasses import fields

from pixeltable.metadata import schema


class MetadataUtils:
    @classmethod
    def _diff_md(
        cls, old_md: dict[int, schema.SchemaColumn] | None, new_md: dict[int, schema.SchemaColumn] | None
    ) -> str:
        """Return a string reporting the differences in a specific entry in two dictionaries

        Results are formatted as follows:
        - If `old_md` is `None`, returns 'Initial Version'.
        - If `old_md` and `new_md` are the same, returns an empty string.
        - If there are additions, changes, or deletions, returns a string summarizing the changes.
        """
        assert new_md is not None
        if old_md is None:
            return 'Initial Version'
        if old_md == new_md:
            return ''
        # Tracks whether any system columns (e.g. index columns) differ between the two versions
        system_columns_differ = False
        # added, altered, and dropped track diffs of user-visible columns only. That's what we want to report to users
        # in the end.
        added, dropped = [], []
        altered = {}

        for col_id, new_col in new_md.items():
            if col_id not in old_md:
                system_columns_differ |= new_col.name is None
                if new_col.name is not None:
                    added.append(new_col.name)
            else:
                old_col = old_md[col_id]
                diff = cls._diff_col(old_col, new_col)
                if diff:
                    if old_col.name is not None:
                        assert new_col.name is not None, "A user-visible column can't become a system column"
                        altered[old_col.name] = diff
                    else:
                        assert new_col.name is None, "A system column can't become user-visible"
                        system_columns_differ = True

        for col_id, old_col in old_md.items():
            if col_id in new_md:
                continue
            system_columns_differ |= old_col.name is None
            if old_col.name is not None:
                dropped.append(old_col.name)

        user_visible_changes = len(added) > 0 or len(altered) > 0 or len(dropped) > 0
        if not user_visible_changes:
            if system_columns_differ:
                # Currently this shouldn't happen, but if in the future we start supporting some kind of schema
                # change that only involves system columns, we'll need to implement a user-friendly way to report
                # it here.
                raise AssertionError(
                    'System-only schema evolution without user-visible changes are not currently supported'
                )
            return ''

        # Format the result
        t = []
        if added:
            t.append('Added: ' + ', '.join(added))
        if altered:
            t.append('Altered: ' + ', '.join((f'{name} ({desc})' for name, desc in altered.items())))
        if dropped:
            t.append('Dropped: ' + ', '.join(dropped))
        return ', '.join(t)

    @classmethod
    def _diff_col(cls, old: schema.SchemaColumn, new: schema.SchemaColumn) -> str | None:
        """Compares two SchemaColumn objects and returns a string describing the differences, or None if they are
        the same.
        """
        assert len(fields(old)) == 7, 'This method needs to be updated whenever SchemaColumn changes'
        diff = []
        # Note: we ignore pos because columns changing places are not very interesting to users, and because they are
        # usually a side effect of other changes such as drop column.
        if old.name != new.name:
            diff.append(f'renamed to {new.name}')
        assert old.is_pk == new.is_pk, 'Not implemented: describe a primary key change'
        assert old.col_type == new.col_type, 'Not implemented: describe a column type change'
        assert old.value_expr == new.value_expr, 'Not implemented: describe a value expression change'
        assert old.media_validation == new.media_validation, 'Not implemented: describe a media validation change'
        assert old.destination == new.destination, 'Not implemented: describe a destination change'

        if diff:
            return ', '.join(diff)
        return None

    @classmethod
    def create_md_change_dict(cls, md_list: list[tuple[int, dict[int, schema.SchemaColumn]]] | None) -> dict[int, str]:
        """Return a dictionary of schema changes by version
        Args:
            md_list: a list of tuples, each containing a version number and a metadata dictionary.
        """
        r: dict[int, str] = {}
        if md_list is None or len(md_list) == 0:
            return r

        # Sort the list in place by version number
        md_list.sort()

        first_retrieved_version = md_list[0][0]
        if first_retrieved_version == 0:
            prev_md = None
            prev_ver = -1
            start = 0
        else:
            prev_md = md_list[0][1]
            prev_ver = first_retrieved_version
            start = 1

        for ver, curr_md in md_list[start:]:
            if ver == prev_ver:
                continue
            assert ver > prev_ver
            tf = cls._diff_md(prev_md, curr_md)
            if tf != '':
                r[ver] = tf
            prev_md = curr_md
        return r
