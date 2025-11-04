from pixeltable.metadata import register_converter

import sqlalchemy as sql

@register_converter(version=41)
def _(engine: sql.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(sql.text("ALTER TABLE dirs ADD COLUMN additional_md JSONB DEFAULT '{}'::JSONB"))
        conn.execute(sql.text("ALTER TABLE tables ADD COLUMN additional_md JSONB DEFAULT '{}'::JSONB"))
        conn.execute(sql.text("ALTER TABLE tableversions ADD COLUMN additional_md JSONB DEFAULT '{}'::JSONB"))
        conn.execute(sql.text("ALTER TABLE tableschemaversions ADD COLUMN additional_md JSONB DEFAULT '{}'::JSONB"))
