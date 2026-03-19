#!/usr/bin/env python
"""
Create a synthetic wardrobe.db for deterministic testing/demo.
This will BACK UP any existing sqlite DB before overwriting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="DATABASE_URL override (e.g. sqlite:///./wardrobe.db)")
    p.add_argument("--tops", type=int, default=16)
    p.add_argument("--bottoms", type=int, default=12)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()

    if args.db:
        os.environ["DATABASE_URL"] = args.db

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from app.database import DATABASE_URL, SessionLocal, Base, engine, ensure_sqlite_schema
    from app.models.user import User
    from app.models.body_profile import BodyProfile
    from app.models.wardrobe import WardrobeItem
    from app.models.outfit import Outfit, OutfitItem
    from app.models.feedback import Feedback

    # Create tables
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_schema()

    # Backup sqlite file if present
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "", 1)
        if os.path.exists(db_path):
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"{db_path}.bak.{ts}"
            shutil.copy2(db_path, backup)
            print(f"Backed up existing db to {backup}")

    rng = random.Random(args.seed)

    def rand_embedding(dim=512):
        vec = [rng.random() for _ in range(dim)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    top_types = ["shirt", "tshirt", "hoodie", "kurta"]
    bottom_types = ["jeans", "trousers", "chinos", "shorts"]
    colors = ["black", "navy", "gray", "white", "beige", "brown", "olive", "blue", "maroon", "khaki"]

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    placeholder = uploads_dir / "synth.png"
    if not placeholder.exists():
        # 1x1 PNG
        placeholder.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D4948445200000001000000010806000000"
                "1F15C4890000000A49444154789C636000000200015A0B2A2B00000000"
                "49454E44AE426082"
            )
        )

    db = SessionLocal()
    try:
        db.query(OutfitItem).delete()
        db.query(Outfit).delete()
        db.query(Feedback).delete()
        db.query(WardrobeItem).delete()
        db.query(BodyProfile).delete()
        db.query(User).delete()
        db.commit()

        user = User(email="synthetic@example.com", password_hash="synthetic")
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(BodyProfile(user_id=user.id))

        def add_item(item_type, category, color, idx):
            item = WardrobeItem(
                user_id=user.id,
                image_url="/uploads/synth.png",
                mask_url=None,
                item_type=item_type,
                category=category,
                color=color,
                color_palette=[color],
                embedding=rand_embedding(),
                suggested_item_type=None,
                suggested_item_type_confidence=None,
                pattern=None,
                fabric=None,
                fit=None,
                size=None,
                season_tags=None,
                brand=None,
                measurements=None,
                confidence_scores=None,
                wear_count=0,
                last_worn_at=None,
                is_active=True,
            )
            db.add(item)

        for i in range(args.tops):
            add_item(rng.choice(top_types), "top", rng.choice(colors), i + 1)
        for i in range(args.bottoms):
            add_item(rng.choice(bottom_types), "bottom", rng.choice(colors), i + 1)

        db.commit()
        print(f"Seeded {args.tops} tops and {args.bottoms} bottoms into {DATABASE_URL}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
