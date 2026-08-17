from __future__ import annotations

from datetime import date
from typing import Any

import requests

SF_GIANTS_TEAM_ID = 137
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
UA = "Mozilla/5.0 GiantsNewsBotV2Schedule/1.0"
TIMEOUT = 12

TEAM_NAME_TO_CANONICAL = {
    "Arizona Diamondbacks": "diamondbacks",
    "Athletics": "athletics",
    "Atlanta Braves": "braves",
    "Baltimore Orioles": "orioles",
    "Boston Red Sox": "red-sox",
    "Chicago Cubs": "cubs",
    "Chicago White Sox": "white-sox",
    "Cincinnati Reds": "reds",
    "Cleveland Guardians": "guardians",
    "Colorado Rockies": "rockies",
    "Detroit Tigers": "tigers",
    "Houston Astros": "astros",
    "Kansas City Royals": "royals",
    "Los Angeles Angels": "angels",
    "Los Angeles Dodgers": "dodgers",
    "Miami Marlins": "marlins",
    "Milwaukee Brewers": "brewers",
    "Minnesota Twins": "twins",
    "New York Mets": "mets",
    "New York Yankees": "yankees",
    "Philadelphia Phillies": "phillies",
    "Pittsburgh Pirates": "pirates",
    "San Diego Padres": "padres",
    "Seattle Mariners": "mariners",
    "St. Louis Cardinals": "cardinals",
    "Tampa Bay Rays": "rays",
    "Texas Rangers": "rangers",
    "Toronto Blue Jays": "blue-jays",
    "Washington Nationals": "nationals",
}


def _opponent_for_game(game: dict[str, Any]) -> tuple[int, str]:
    teams = game.get("teams", {}) if isinstance(game, dict) else {}
    for side in ("away", "home"):
        team = ((teams.get(side) or {}).get("team") or {})
        team_id = int(team.get("id") or 0)
        if team_id and team_id != SF_GIANTS_TEAM_ID:
            name = str(team.get("name") or "")
            return team_id, TEAM_NAME_TO_CANONICAL.get(name, "")
    return 0, ""


def fetch_giants_schedule(
    start_date: date,
    end_date: date,
    *,
    timeout: int = TIMEOUT,
    session: requests.Session | None = None,
) -> list[dict]:
    """Return Giants games from MLB StatsAPI. Failure is non-blocking upstream."""
    client = session or requests
    response = client.get(
        STATSAPI_SCHEDULE_URL,
        params={
            "sportId": 1,
            "teamId": SF_GIANTS_TEAM_ID,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        },
        headers={"User-Agent": UA},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    games: list[dict] = []
    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            opponent_id, opponent = _opponent_for_game(game)
            status = game.get("status", {}) or {}
            games.append({
                "game_pk": int(game.get("gamePk") or 0),
                "official_date": str(game.get("officialDate") or day.get("date") or ""),
                "game_date": str(game.get("gameDate") or ""),
                "opponent_id": opponent_id,
                "opponent": opponent,
                "abstract_state": str(status.get("abstractGameState") or ""),
                "detailed_state": str(status.get("detailedState") or ""),
            })
    return games
