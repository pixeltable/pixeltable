import uuid

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import SystemInfo


@register_converter(version=56)
def _(conn: sql.Connection) -> None:
    """
    Changes in version 57:
    - SystemInfoMd has an install_id.
    """
    md = conn.execute(sql.select(SystemInfo.md)).scalar_one()
    conn.execute(sql.update(SystemInfo).values(md={**md, 'install_id': uuid.uuid4().hex}))
