from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def migrate_legacy_schema():
    """Add fields introduced by the current dashboard to older SQLite databases."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    additions = {
        "students": {
            "lcid": "VARCHAR(50)",
            "profile_image_path": "VARCHAR(255)",
            "face_image_path": "VARCHAR(255)",
            "face_encoding": "TEXT",
        },
        "attendances": {"verification_status": "VARCHAR(20)"},
    }
    with engine.begin() as connection:
        for table, columns in additions.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, column_type in columns.items():
                if name not in existing:
                    connection.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")
                    )
        student_columns = {column["name"] for column in inspector.get_columns("students")}
        if "roll_number" in student_columns:
            connection.execute(
                text(
                    "UPDATE students SET lcid = roll_number "
                    "WHERE lcid IS NULL AND roll_number IS NOT NULL"
                )
            )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()