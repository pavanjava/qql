from datasets import load_dataset
from datetime import date

# ── config ────────────────────────────────────────────────────────────────────
REPO_ID     = "pavanmantha/doctor_patient_conversation"
COLLECTION  = "doctor_patient_conversation"
OUTPUT_FILE = "data_sets/source_data.qql"
TODAY       = date.today().isoformat()   # e.g. 2026-05-16
BATCH_SIZE  = 200                        # rows per INSERT block (stays under 33MB)
# ─────────────────────────────────────────────────────────────────────────────


def escape(text: str) -> str:
    """Escape single-quotes inside field values."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def build_record(row: dict) -> str:
    description = escape((row.get("description") or "").strip())
    text        = escape((row.get("conversation") or "").strip())
    status      = escape((row.get("status")       or "").strip())

    return (
        "  {\n"
        f"    'description': '{description}',\n"
        f"    'text': '{text}',\n"
        f"    'status': '{status}'\n"
        "  }"
    )


def write_batch(f, batch: list, batch_num: int, collection: str):
    f.write(f"\n-- Batch {batch_num} ({len(batch)} records)\n")
    f.write(f"INSERT BULK INTO COLLECTION {collection} VALUES [\n")
    for i, record in enumerate(batch):
        is_last = (i == len(batch) - 1)
        f.write(record)
        f.write("\n" if is_last else ",\n")
    f.write("]\n")


def main():
    print(f"Loading dataset from '{REPO_ID}' ...")
    ds = load_dataset(REPO_ID, split="train")
    total = len(ds)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  -> {total} rows loaded")
    print(f"  -> {BATCH_SIZE} rows per INSERT block")
    print(f"  -> {num_batches} total batches\n")

    header = f"""\
-- Qdrant Query Language
-- BULK INSERT -- DOCTOR PATIENT CONVERSATION (BATCHED)

-- ============================================================
--  QQL -- Doctor Patient Conversation
--  Collection : {COLLECTION}
--  Source     : {REPO_ID}
--  Total rows : {total}
--  Batch size : {BATCH_SIZE}
--  Generated  : {TODAY}
-- ============================================================

-- Step 0: Show Collections
SHOW COLLECTIONS

-- Step 1: Create the collection
CREATE COLLECTION {COLLECTION}

-- ============================================================
-- BULK INSERT -- BATCHED (each block <= {BATCH_SIZE} records)
-- ============================================================
"""

    print(f"Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header)

        batch     = []
        batch_num = 1

        for i, row in enumerate(ds):
            batch.append(build_record(row))

            if len(batch) == BATCH_SIZE:
                write_batch(f, batch, batch_num, COLLECTION)
                print(f"  batch {batch_num} written ({i + 1}/{total} rows)")
                batch     = []
                batch_num += 1

        # flush remaining rows
        if batch:
            write_batch(f, batch, batch_num, COLLECTION)
            print(f"  batch {batch_num} written ({total}/{total} rows)")

    print(f"\nDone. '{OUTPUT_FILE}' written -- {total} records across {batch_num} INSERT blocks.")


if __name__ == "__main__":
    main()