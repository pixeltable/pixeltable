from __future__ import annotations

from typing import Optional

from pixeltable.metadata import schema


class MetadataUtils:
    @classmethod
    def _diff_md(
        cls, old_md: Optional[dict[int, schema.SchemaColumn]], new_md: Optional[dict[int, schema.SchemaColumn]]
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
        added = {k: v.name for k, v in new_md.items() if k not in old_md}
        changed = {
            k: f'{old_md[k].name!r} to {v.name!r}'
            for k, v in new_md.items()
            if k in old_md and old_md[k].name != v.name
        }
        deleted = {k: v.name for k, v in old_md.items() if k not in new_md}
        if len(added) == 0 and len(changed) == 0 and len(deleted) == 0:
            return ''
        # Format the result
        t = []
        if len(added) > 0:
            t.append('Added: ' + ', '.join(added.values()))
        if len(changed) > 0:
            t.append('Renamed: ' + ', '.join(changed.values()))
        if len(deleted) > 0:
            t.append('Deleted: ' + ', '.join(deleted.values()))
        r = ', '.join(t)
        return r

    @classmethod
    def _create_md_change_dict(
        cls, md_list: Optional[list[tuple[int, dict[int, schema.SchemaColumn]]]]
    ) -> dict[int, str]:
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
