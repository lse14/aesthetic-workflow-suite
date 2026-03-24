import sqlite3
import threading
from pathlib import Path
from typing import Any


class AnnotationDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_post_id TEXT,
                    source_page_url TEXT,
                    original_url TEXT,
                    local_path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_source_post
                ON samples(source, source_post_id)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    sample_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('labeled','skipped')),
                    aesthetic INTEGER,
                    composition INTEGER,
                    color INTEGER,
                    sexual INTEGER,
                    in_domain INTEGER NOT NULL DEFAULT 1,
                    content_type TEXT NOT NULL DEFAULT 'anime_illust',
                    exclude_from_train INTEGER NOT NULL DEFAULT 0,
                    exclude_from_score_train INTEGER NOT NULL DEFAULT 0,
                    exclude_from_cls_train INTEGER NOT NULL DEFAULT 0,
                    exclude_reason TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE,
                    CHECK(aesthetic BETWEEN 1 AND 5 OR aesthetic IS NULL),
                    CHECK(composition BETWEEN 1 AND 5 OR composition IS NULL),
                    CHECK(color BETWEEN 1 AND 5 OR color IS NULL),
                    CHECK(sexual BETWEEN 1 AND 5 OR sexual IS NULL),
                    CHECK(in_domain IN (0,1)),
                    CHECK(exclude_from_train IN (0,1)),
                    CHECK(exclude_from_score_train IN (0,1)),
                    CHECK(exclude_from_cls_train IN (0,1))
                )
                """
            )
            self._ensure_annotation_columns(cur)
            self.conn.commit()

    @staticmethod
    def _has_column(cur: sqlite3.Cursor, table: str, column: str) -> bool:
        rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
        names = {r[1] for r in rows}
        return column in names

    def _ensure_annotation_columns(self, cur: sqlite3.Cursor) -> None:
        # Lightweight migration for existing DBs.
        if not self._has_column(cur, "annotations", "in_domain"):
            cur.execute("ALTER TABLE annotations ADD COLUMN in_domain INTEGER NOT NULL DEFAULT 1")
        if not self._has_column(cur, "annotations", "content_type"):
            cur.execute(
                "ALTER TABLE annotations ADD COLUMN content_type TEXT NOT NULL DEFAULT 'anime_illust'"
            )
        if not self._has_column(cur, "annotations", "exclude_from_train"):
            cur.execute(
                "ALTER TABLE annotations ADD COLUMN exclude_from_train INTEGER NOT NULL DEFAULT 0"
            )
        if not self._has_column(cur, "annotations", "exclude_from_score_train"):
            cur.execute(
                "ALTER TABLE annotations ADD COLUMN exclude_from_score_train INTEGER NOT NULL DEFAULT 0"
            )
        if not self._has_column(cur, "annotations", "exclude_from_cls_train"):
            cur.execute(
                "ALTER TABLE annotations ADD COLUMN exclude_from_cls_train INTEGER NOT NULL DEFAULT 0"
            )
        if not self._has_column(cur, "annotations", "exclude_reason"):
            cur.execute("ALTER TABLE annotations ADD COLUMN exclude_reason TEXT")
        if not self._has_column(cur, "annotations", "note"):
            cur.execute("ALTER TABLE annotations ADD COLUMN note TEXT")
        self._drop_background_column_if_exists(cur)

        # Backfill split flags from legacy flag only for clearly unmigrated rows.
        # Important: do not overwrite rows that were already manually split
        # (e.g. score=1, cls=0), otherwise startup would "reset" user edits.
        if self._has_column(cur, "annotations", "exclude_from_train"):
            cur.execute(
                """
                UPDATE annotations
                SET exclude_from_score_train = 1,
                    exclude_from_cls_train = 1
                WHERE COALESCE(exclude_from_train, 0) = 1
                  AND COALESCE(exclude_from_score_train, 0) = 0
                  AND COALESCE(exclude_from_cls_train, 0) = 0
                """
            )
            cur.execute(
                """
                UPDATE annotations
                SET exclude_from_train = CASE
                    WHEN COALESCE(exclude_from_score_train, 0) = 1
                         OR COALESCE(exclude_from_cls_train, 0) = 1
                    THEN 1 ELSE 0 END
                """
            )

    def _drop_background_column_if_exists(self, cur: sqlite3.Cursor) -> None:
        if not self._has_column(cur, "annotations", "background"):
            return
        cur.execute("PRAGMA foreign_keys=OFF")
        cur.execute(
            """
            CREATE TABLE annotations_new (
                sample_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('labeled','skipped')),
                aesthetic INTEGER,
                composition INTEGER,
                color INTEGER,
                sexual INTEGER,
                in_domain INTEGER NOT NULL DEFAULT 1,
                content_type TEXT NOT NULL DEFAULT 'anime_illust',
                exclude_from_train INTEGER NOT NULL DEFAULT 0,
                exclude_from_score_train INTEGER NOT NULL DEFAULT 0,
                exclude_from_cls_train INTEGER NOT NULL DEFAULT 0,
                exclude_reason TEXT,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE,
                CHECK(aesthetic BETWEEN 1 AND 5 OR aesthetic IS NULL),
                CHECK(composition BETWEEN 1 AND 5 OR composition IS NULL),
                CHECK(color BETWEEN 1 AND 5 OR color IS NULL),
                CHECK(sexual BETWEEN 1 AND 5 OR sexual IS NULL),
                CHECK(in_domain IN (0,1)),
                CHECK(exclude_from_train IN (0,1)),
                CHECK(exclude_from_score_train IN (0,1)),
                CHECK(exclude_from_cls_train IN (0,1))
            )
            """
        )
        cur.execute(
            """
            INSERT INTO annotations_new (
                sample_id, status, aesthetic, composition, color, sexual,
                in_domain, content_type, exclude_from_train, exclude_from_score_train,
                exclude_from_cls_train, exclude_reason, note, created_at, updated_at
            )
            SELECT
                sample_id,
                status,
                aesthetic,
                composition,
                color,
                sexual,
                COALESCE(in_domain, 1),
                COALESCE(content_type, 'anime_illust'),
                COALESCE(exclude_from_train, 0),
                COALESCE(exclude_from_score_train, COALESCE(exclude_from_train, 0)),
                COALESCE(exclude_from_cls_train, COALESCE(exclude_from_train, 0)),
                exclude_reason,
                note,
                COALESCE(created_at, CURRENT_TIMESTAMP),
                COALESCE(updated_at, CURRENT_TIMESTAMP)
            FROM annotations
            """
        )
        cur.execute("DROP TABLE annotations")
        cur.execute("ALTER TABLE annotations_new RENAME TO annotations")
        cur.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def get_sample_by_id(self, sample_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_annotation_by_sample_id(self, sample_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM annotations WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_sample_with_annotation(self, sample_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                """
                SELECT
                    s.*,
                    a.status AS ann_status,
                    a.aesthetic AS ann_aesthetic,
                    a.composition AS ann_composition,
                    a.color AS ann_color,
                    a.sexual AS ann_sexual,
                    a.in_domain AS ann_in_domain,
                    a.content_type AS ann_content_type,
                    a.exclude_from_train AS ann_exclude_from_train,
                    a.exclude_from_score_train AS ann_exclude_from_score_train,
                    a.exclude_from_cls_train AS ann_exclude_from_cls_train,
                    a.exclude_reason AS ann_exclude_reason,
                    a.note AS ann_note,
                    a.updated_at AS ann_updated_at
                FROM samples s
                LEFT JOIN annotations a ON a.sample_id = s.id
                WHERE s.id = ?
                """,
                (sample_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_sample_position(self, sample_id: int) -> dict[str, int] | None:
        with self.lock:
            exists = self.conn.execute(
                "SELECT 1 FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not exists:
                return None
            row = self.conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM samples WHERE id <= ?) AS pos,
                    (SELECT COUNT(*) FROM samples) AS total
                """,
                (sample_id,),
            ).fetchone()
        return {
            "position": int(row["pos"] or 0),
            "total": int(row["total"] or 0),
        }

    def get_last_reviewed_sample(self, *, status: str | None = None) -> dict[str, Any] | None:
        where_sql = ""
        params: list[Any] = []
        st = (status or "").strip().lower()
        if st in {"labeled", "skipped"}:
            where_sql = "WHERE a.status = ?"
            params.append(st)

        with self.lock:
            row = self.conn.execute(
                f"""
                SELECT
                    s.*,
                    a.status AS ann_status,
                    a.aesthetic AS ann_aesthetic,
                    a.composition AS ann_composition,
                    a.color AS ann_color,
                    a.sexual AS ann_sexual,
                    a.in_domain AS ann_in_domain,
                    a.content_type AS ann_content_type,
                    a.exclude_from_train AS ann_exclude_from_train,
                    a.exclude_from_score_train AS ann_exclude_from_score_train,
                    a.exclude_from_cls_train AS ann_exclude_from_cls_train,
                    a.exclude_reason AS ann_exclude_reason,
                    a.note AS ann_note,
                    a.updated_at AS ann_updated_at
                FROM annotations a
                JOIN samples s ON s.id = a.sample_id
                {where_sql}
                ORDER BY a.updated_at DESC, a.sample_id DESC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
        return self._row_to_dict(row)

    def get_sample_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM samples WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_sample_by_source_post(
        self,
        source: str,
        source_post_id: str | None,
    ) -> dict[str, Any] | None:
        if source_post_id is None:
            return None
        with self.lock:
            if str(source).lower() == "local":
                # Local source path matching should be case-insensitive on Windows.
                row = self.conn.execute(
                    "SELECT * FROM samples WHERE source = ? AND lower(source_post_id) = lower(?)",
                    (source, source_post_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM samples WHERE source = ? AND source_post_id = ?",
                    (source, source_post_id),
                ).fetchone()
        return self._row_to_dict(row)

    def is_reviewed(self, sample_id: int) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM annotations WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
        return row is not None

    def insert_sample(
        self,
        *,
        source: str,
        source_post_id: str | None,
        source_page_url: str | None,
        original_url: str | None,
        local_path: str,
        sha256: str,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO samples(
                    source, source_post_id, source_page_url, original_url,
                    local_path, sha256, width, height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    source_post_id,
                    source_page_url,
                    original_url,
                    local_path,
                    sha256,
                    width,
                    height,
                ),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM samples WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        out = self._row_to_dict(row)
        if out is None:
            raise RuntimeError("Failed to insert sample.")
        return out

    def upsert_label(
        self,
        *,
        sample_id: int,
        aesthetic: int | None,
        composition: int | None,
        color: int | None,
        sexual: int | None,
        in_domain: int,
        content_type: str,
        exclude_from_score_train: int,
        exclude_from_cls_train: int,
        exclude_reason: str | None,
        status: str,
        note: str | None = None,
    ) -> None:
        legacy_exclude = (
            1 if int(exclude_from_score_train) == 1 or int(exclude_from_cls_train) == 1 else 0
        )
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO annotations(
                    sample_id, status, aesthetic, composition, color, sexual,
                    in_domain, content_type, exclude_from_train,
                    exclude_from_score_train, exclude_from_cls_train, exclude_reason, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    status = excluded.status,
                    aesthetic = excluded.aesthetic,
                    composition = excluded.composition,
                    color = excluded.color,
                    sexual = excluded.sexual,
                    in_domain = excluded.in_domain,
                    content_type = excluded.content_type,
                    exclude_from_train = excluded.exclude_from_train,
                    exclude_from_score_train = excluded.exclude_from_score_train,
                    exclude_from_cls_train = excluded.exclude_from_cls_train,
                    exclude_reason = excluded.exclude_reason,
                    note = excluded.note,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    sample_id,
                    status,
                    aesthetic,
                    composition,
                    color,
                    sexual,
                    in_domain,
                    content_type,
                    legacy_exclude,
                    int(exclude_from_score_train),
                    int(exclude_from_cls_train),
                    exclude_reason,
                    note,
                ),
            )
            self.conn.commit()

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.conn.execute("SELECT COUNT(*) AS n FROM samples").fetchone()["n"]
            labeled = self.conn.execute(
                "SELECT COUNT(*) AS n FROM annotations WHERE status='labeled'"
            ).fetchone()["n"]
            skipped = self.conn.execute(
                "SELECT COUNT(*) AS n FROM annotations WHERE status='skipped'"
            ).fetchone()["n"]
            by_source_rows = self.conn.execute(
                """
                SELECT s.source AS source, COUNT(*) AS n
                FROM annotations a
                JOIN samples s ON s.id = a.sample_id
                WHERE a.status = 'labeled'
                GROUP BY s.source
                ORDER BY n DESC
                """
            ).fetchall()
        by_source = {r["source"]: r["n"] for r in by_source_rows}
        return {
            "total_samples": int(total),
            "labeled_samples": int(labeled),
            "skipped_samples": int(skipped),
            "unreviewed_samples": int(total - labeled - skipped),
            "labeled_by_source": by_source,
        }

    def iter_labeled_rows(self):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT
                    s.id AS id,
                    s.source AS source,
                    s.source_post_id AS source_post_id,
                    s.source_page_url AS source_page_url,
                    s.local_path AS local_path,
                    a.aesthetic AS aesthetic,
                    a.composition AS composition,
                    a.color AS color,
                    a.sexual AS sexual,
                    a.in_domain AS in_domain,
                    a.content_type AS content_type,
                    a.exclude_from_train AS exclude_from_train,
                    a.exclude_from_score_train AS exclude_from_score_train,
                    a.exclude_from_cls_train AS exclude_from_cls_train,
                    a.exclude_reason AS exclude_reason
                FROM samples s
                JOIN annotations a ON a.sample_id = s.id
                WHERE a.status = 'labeled'
                ORDER BY s.id ASC
                """
            ).fetchall()
        for row in rows:
            yield self._row_to_dict(row)

    def list_sources(self) -> list[str]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT DISTINCT source
                FROM samples
                ORDER BY source ASC
                """
            ).fetchall()
        return [str(r["source"]) for r in rows if r["source"]]

    def list_samples(
        self,
        *,
        page: int,
        size: int,
        status: str,
        source: str | None = None,
        order: str = "desc",
        in_domain: int | None = None,
        content_type: str | None = None,
        score_dim: str | None = None,
        score_value: int | None = None,
        after_id: int | None = None,
    ) -> dict[str, Any]:
        offset = (page - 1) * size
        where = []
        params: list[Any] = []
        if status == "labeled":
            where.append("a.status = 'labeled'")
        elif status == "skipped":
            where.append("a.status = 'skipped'")
        elif status == "unreviewed":
            where.append("a.sample_id IS NULL")
        if source:
            where.append("s.source = ?")
            params.append(source)
        if in_domain is not None:
            where.append("a.in_domain = ?")
            params.append(int(in_domain))
        if content_type:
            where.append("a.content_type = ?")
            params.append(str(content_type))
        if score_dim and score_value is not None:
            dim_to_col = {
                "aesthetic": "a.aesthetic",
                "composition": "a.composition",
                "color": "a.color",
                "sexual": "a.sexual",
            }
            col = dim_to_col.get(score_dim)
            if col:
                where.append(f"{col} = ?")
                params.append(int(score_value))
        if after_id is not None:
            where.append("s.id > ?")
            params.append(int(after_id))

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        order_sql = "ASC" if str(order).strip().lower() == "asc" else "DESC"
        with self.lock:
            total = self.conn.execute(
                f"""
                SELECT COUNT(*) AS n
                FROM samples s
                LEFT JOIN annotations a ON a.sample_id = s.id
                {where_sql}
                """,
                tuple(params),
            ).fetchone()["n"]

            rows = self.conn.execute(
                f"""
                SELECT
                    s.*,
                    a.status AS ann_status,
                    a.aesthetic AS ann_aesthetic,
                    a.composition AS ann_composition,
                    a.color AS ann_color,
                    a.sexual AS ann_sexual,
                    a.in_domain AS ann_in_domain,
                    a.content_type AS ann_content_type,
                    a.exclude_from_train AS ann_exclude_from_train,
                    a.exclude_from_score_train AS ann_exclude_from_score_train,
                    a.exclude_from_cls_train AS ann_exclude_from_cls_train,
                    a.exclude_reason AS ann_exclude_reason,
                    a.note AS ann_note,
                    a.updated_at AS ann_updated_at
                FROM samples s
                LEFT JOIN annotations a ON a.sample_id = s.id
                {where_sql}
                ORDER BY s.id {order_sql}
                LIMIT ? OFFSET ?
                """,
                tuple(params + [size, offset]),
            ).fetchall()

        items = [self._row_to_dict(r) for r in rows]
        return {"total": int(total), "items": items}

    def list_unreviewed_after(
        self,
        *,
        after_sample_id: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        after_id = int(after_sample_id)
        limit_v = max(1, int(limit))
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT s.*
                FROM samples s
                LEFT JOIN annotations a ON a.sample_id = s.id
                WHERE a.sample_id IS NULL
                  AND s.id > ?
                ORDER BY s.id ASC
                LIMIT ?
                """,
                (after_id, limit_v),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = self._row_to_dict(row)
            if item is not None:
                out.append(item)
        return out

    def delete_sample(self, sample_id: int) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "DELETE FROM samples WHERE id = ?",
                (sample_id,),
            )
            self.conn.commit()
        return int(cur.rowcount or 0) > 0

    def close(self) -> None:
        with self.lock:
            self.conn.close()
