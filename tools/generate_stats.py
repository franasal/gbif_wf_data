#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

GERMAN_STATES = {
    "baden-württemberg": "Baden-Württemberg",
    "bayern": "Bayern",
    "berlin": "Berlin",
    "brandenburg": "Brandenburg",
    "bremen": "Bremen",
    "hamburg": "Hamburg",
    "hessen": "Hessen",
    "mecklenburg-vorpommern": "Mecklenburg-Vorpommern",
    "niedersachsen": "Niedersachsen",
    "nordrhein-westfalen": "Nordrhein-Westfalen",
    "rheinland-pfalz": "Rheinland-Pfalz",
    "saarland": "Saarland",
    "sachsen": "Sachsen",
    "sachsen-anhalt": "Sachsen-Anhalt",
    "schleswig-holstein": "Schleswig-Holstein",
    "thüringen": "Thüringen",
}

def norm_state(s: str) -> str:
    if not s:
        return "Unknown"
    t = s.strip().lower()
    t = t.replace("ue", "ü").replace("ae", "ä").replace("oe", "ö")
    t = t.replace(".", "").replace(",", " ").replace("  ", " ").strip()
    t = t.replace("nrw", "nordrhein-westfalen")
    t = t.replace("thueringen", "thüringen")
    t = t.replace("mecklenburg vorpommern", "mecklenburg-vorpommern")
    t = t.replace("schleswig holstein", "schleswig-holstein")
    return GERMAN_STATES.get(t, s.strip())

def main():
    ap = argparse.ArgumentParser(description="Generate summary stats (per-plant + per state).")
    ap.add_argument("--db", required=True)
    ap.add_argument("--occ-json", required=True, help="occurrences_compact.json or .gz already unpacked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="DE")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--top-plants-per-state", type=int, default=30)

    args = ap.parse_args()

    # Load exported plants list
    occ_path = Path(args.occ_json)
    data = json.loads(occ_path.read_text(encoding="utf-8"))
    plants = list((data.get("plants") or {}).keys())
    if not plants:
        raise SystemExit("No plants found in occ json.")

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    # Build WHERE consistent with pipeline
    where = ["lat IS NOT NULL", "lon IS NOT NULL"]
    params: List[object] = []

    if args.country:
        where.append("countryCode = ?")
        params.append(args.country)

    if args.year_from is not None:
        where.append("year >= ?")
        params.append(args.year_from)
    if args.year_to is not None:
        where.append("year <= ?")
        params.append(args.year_to)

    where_sql = "WHERE " + " AND ".join(where)

    # Per plant: true totals already in export; compute share and sampled share
    total_true_all = 0
    total_sampled_all = 0
    per_plant = {}

    for sci in plants:
        obj = data["plants"][sci]
        total_true_all += int(obj.get("total") or 0)
        total_sampled_all += int(obj.get("sampled_total") or len(obj.get("points") or []))

    for sci in plants:
        obj = data["plants"][sci]
        ttrue = int(obj.get("total") or 0)
        tsamp = int(obj.get("sampled_total") or len(obj.get("points") or []))
        per_plant[sci] = {
            "true_total": ttrue,
            "sampled_total": tsamp,
            "true_share": (ttrue / total_true_all) if total_true_all else 0.0,
            "sampled_share": (tsamp / total_sampled_all) if total_sampled_all else 0.0,
        }

    # Per state: top plants by TRUE counts in DB (needs stateProvince column)
    # Use temp table for plant filter
    con.execute("DROP TABLE IF EXISTS _wanted_species;")
    con.execute("CREATE TEMP TABLE _wanted_species (sci TEXT PRIMARY KEY);")
    con.executemany("INSERT OR IGNORE INTO _wanted_species(sci) VALUES (?)", [(p,) for p in plants])
    con.commit()

    q_state = f"""
      SELECT
        COALESCE(stateProvince, '') AS st,
        species AS sci,
        COUNT(1) AS n
      FROM occ
      {where_sql}
      AND species IN (SELECT sci FROM _wanted_species)
      GROUP BY st, sci
    """
    rows = con.execute(q_state, params).fetchall()

    by_state: Dict[str, List[Tuple[str,int]]] = {}
    for r in rows:
        st = norm_state(r["st"] or "")
        sci = r["sci"]
        n = int(r["n"])
        by_state.setdefault(st, []).append((sci, n))

    # sort and cut
    state_top = {}
    for st, items in by_state.items():
        items.sort(key=lambda x: x[1], reverse=True)
        state_top[st] = [{"sci": sci, "count": n} for sci, n in items[: int(args.top_plants_per_state)]]

    con.close()

    out = {
        "meta": {
            "country": args.country,
            "year_from": args.year_from,
            "year_to": args.year_to,
            "plants_count": len(plants),
            "true_total_all": total_true_all,
            "sampled_total_all": total_sampled_all
        },
        "per_plant": per_plant,
        "top_plants_by_state": state_top
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", str(out_path))

if __name__ == "__main__":
    main()
