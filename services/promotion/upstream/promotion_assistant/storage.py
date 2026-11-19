"""SQLite metadata + immutable PNGs. Keep this directory outside Drive sync."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import uuid


class NotFound(ValueError):
    pass


class Conflict(ValueError):
    pass


class Repository:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "posters.sqlite3"
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS projects(
                id TEXT PRIMARY KEY, store_id TEXT NOT NULL, brief TEXT NOT NULL,
                photo BLOB, logo BLOB, style TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS drafts(
                id TEXT NOT NULL, version INTEGER NOT NULL, project_id TEXT NOT NULL,
                record TEXT NOT NULL, image_name TEXT UNIQUE NOT NULL,
                PRIMARY KEY(id,version), FOREIGN KEY(project_id) REFERENCES projects(id));
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(projects)")}
            if "cutout" not in columns:
                db.execute("ALTER TABLE projects ADD COLUMN cutout BLOB")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _save_image(self, png):
        name = uuid.uuid4().hex + ".png"
        destination = self.images / name
        with destination.open("xb") as stream:
            stream.write(png)
        return name

    def create(self, brief, photo, logo, style, records_and_pngs, *, cutout=None):
        project = uuid.uuid4().hex
        created = []
        try:
            with self.connect() as db:
                db.execute("INSERT INTO projects(id,store_id,brief,photo,logo,style,cutout) VALUES(?,?,?,?,?,?,?)",
                           (project, brief.store.store_id, json.dumps(brief.facts(), ensure_ascii=False),
                            photo, logo, style, cutout))
                for record, png in records_and_pngs:
                    name = self._save_image(png)
                    created.append(name)
                    record.update(image_name=name, project_id=project)
                    db.execute("INSERT INTO drafts VALUES(?,?,?,?,?)",
                               (record["id"], record["version"], project, json.dumps(record, ensure_ascii=False), name))
            return [record for record, _ in records_and_pngs]
        except Exception:
            for name in created:
                (self.images / name).unlink(missing_ok=True)
            raise

    def get(self, draft_id, store_id):
        with self.connect() as db:
            row = db.execute("""SELECT d.record,p.brief,COALESCE(p.cutout,p.photo) AS photo,p.logo,p.style FROM drafts d
                JOIN projects p ON d.project_id=p.id WHERE d.id=? AND p.store_id=?
                ORDER BY d.version DESC LIMIT 1""", (draft_id, store_id)).fetchone()
        if not row:
            raise NotFound("시안을 찾을 수 없습니다.")
        return json.loads(row["record"]), json.loads(row["brief"]), row["photo"], row["logo"], row["style"]

    def revise(self, record, png, expected):
        name = None
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                latest = db.execute("SELECT MAX(version) FROM drafts WHERE id=?", (record["id"],)).fetchone()[0]
                if latest != expected:
                    raise Conflict("다른 수정이 먼저 저장됐습니다. 최신 시안을 다시 열어 주세요.")
                name = self._save_image(png)
                record["image_name"] = name
                db.execute("INSERT INTO drafts VALUES(?,?,?,?,?)",
                           (record["id"], record["version"], record["project_id"],
                            json.dumps(record, ensure_ascii=False), name))
            return record
        except Exception:
            if name:
                (self.images / name).unlink(missing_ok=True)
            raise

    def image(self, name):
        if not name.endswith(".png") or len(name) != 36:
            raise NotFound("이미지를 찾을 수 없습니다.")
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM drafts WHERE image_name=?", (name,)).fetchone()
        path = self.images / name
        if not exists or not path.is_file():
            raise NotFound("이미지를 찾을 수 없습니다.")
        return path
