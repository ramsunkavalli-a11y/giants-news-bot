from __future__ import annotations

import json
from datetime import date

from v2_mlb_schedule import fetch_giants_schedule


def main() -> None:
    games = fetch_giants_schedule(date(2026, 8, 16), date(2026, 8, 16))
    rockies = [game for game in games if game.get("opponent") == "rockies"]
    if not rockies or not rockies[0].get("game_pk"):
        raise RuntimeError("MLB StatsAPI probe did not return the Aug. 16 Giants-Rockies game")
    payload = {"games": games}
    with open("v2-schedule-probe.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    game = rockies[0]
    print(
        f"MLB_SCHEDULE game_pk={game['game_pk']} official_date={game['official_date']} "
        f"opponent={game['opponent']} state={game['detailed_state']}"
    )


if __name__ == "__main__":
    main()
