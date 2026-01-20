# Aegis Platform - MTFI Data Requirements vs Sportsmonks API Capabilities

## Overview

This document maps the data requirements for Aegis Platform's MTFI (Manager Tactical Fit Intelligence) core outputs against what Sportsmonks API v3 can provide.

---

## 1. Manager DNA Engine Requirements

| MTFI Requirement | Sportsmonks Capability | Endpoint/Include | Risk Level |
|-----------------|----------------------|------------------|------------|
| **Formation tendencies** | ✅ Available | `fixtures?include=formations` | LOW |
| **Match results history** | ✅ Available | `fixtures` by team/date range | LOW |
| **Goals scored/conceded** | ✅ Available | `fixtures?include=scores` | LOW |
| **Possession %** | ✅ Available | `fixtures?include=statistics` (type_id for possession) | LOW |
| **Pressing intensity** | ⚠️ Proxy only | No direct PPDA data; use tackles, interceptions, fouls as proxies | HIGH |
| **Build-up patterns** | ⚠️ Proxy only | Pass completion %, long balls %; no passing sequences | HIGH |
| **Chance creation zones** | ⚠️ Limited | Shots, shots on target; no spatial data | MEDIUM |
| **Defensive shape** | ⚠️ Proxy only | Clean sheets, goals conceded, defensive actions | MEDIUM |
| **Transition behaviours** | ❌ Not available | No counter-attack or transition data | HIGH |
| **xG data** | ✅ Available (recent) | `v3/football/expected/fixtures` | LOW |

### Manager DNA - Achievable Metrics

**High Confidence (directly available):**
- Formation frequency (4-3-3 vs 4-4-2 etc.)
- Win/Draw/Loss ratios
- Goals per game (scored/conceded)
- Possession % average
- xG and xGA (expected goals against)
- Clean sheet %

**Medium Confidence (proxy metrics):**
- Pressing intensity → use tackles + interceptions per game
- Defensive solidity → goals conceded + clean sheets + defensive actions
- Attacking directness → long balls % + crosses %

**Low Confidence (significant limitations):**
- Build-up play style (no passing sequence data)
- Transition speed (no counter-attack metrics)
- Pressing trigger zones (no spatial data)

---

## 2. Squad Fit Analysis Requirements

| MTFI Requirement | Sportsmonks Capability | Endpoint/Include | Risk Level |
|-----------------|----------------------|------------------|------------|
| **Player positions** | ✅ Available | `players?include=position,detailedPosition` | LOW |
| **Player statistics (seasonal)** | ✅ Available | `players?include=statistics.details` | LOW |
| **Physical metrics (height/weight)** | ✅ Available | `players` entity fields | LOW |
| **Preferred foot** | ✅ Available | `players?include=metadata` | LOW |
| **Age/experience** | ✅ Available | `players` entity fields | LOW |
| **Goals/assists** | ✅ Available | Player statistics types | LOW |
| **Pass completion %** | ✅ Available | Player statistics types | LOW |
| **Tackles/interceptions** | ✅ Available | Player statistics types | LOW |
| **Dribbles/duels** | ✅ Available | Player statistics types | LOW |
| **Key passes** | ✅ Available | Player statistics types | LOW |
| **Player ratings** | ✅ Available | Player statistics type 118 (rating) | LOW |
| **Minutes played** | ✅ Available | Player statistics types | LOW |
| **GPS/physical load data** | ❌ Not available | Enterprise data only (Catapult etc.) | N/A |
| **Heatmaps/positional data** | ❌ Not available | Enterprise data only | N/A |

### Squad Profiling - Available Player Statistics (Type IDs)

From Sportsmonks documentation, key player stat types include:
- Goals (45), Assists (79)
- Shots on/off target (41, 42)
- Pass accuracy/total passes
- Tackles, interceptions, clearances
- Dribbles attempted/successful
- Duels won/lost
- Key passes (117)
- Minutes played (119)
- Match rating (118)
- Yellow/red cards

---

## 3. Recruitment Impact Requirements

| MTFI Requirement | Sportsmonks Capability | Endpoint/Include | Risk Level |
|-----------------|----------------------|------------------|------------|
| **Current squad roster** | ✅ Available | `teams/{id}?include=players` | LOW |
| **Player market values** | ⚠️ Limited | May need external source (Transfermarkt) | MEDIUM |
| **Transfer history** | ✅ Available | `players?include=transfers` | LOW |
| **Contract data** | ❌ Not available | External source needed | HIGH |

---

## 4. Key Sportsmonks Endpoints for MTFI

### Core Endpoints

```
# Fixtures (match data)
GET /v3/football/fixtures/between/{start}/{end}
GET /v3/football/fixtures?include=formations,statistics,lineups,scores,participants

# Teams
GET /v3/football/teams/{team_id}?include=players,statistics,coaches

# Players  
GET /v3/football/players/{player_id}?include=statistics.details,position,metadata

# Coaches (Managers)
GET /v3/football/coaches/{coach_id}?include=statistics,teams,fixtures

# Seasons & Leagues
GET /v3/football/seasons/{season_id}?include=fixtures,teams
GET /v3/football/leagues/{league_id}?include=currentSeason

# Expected Goals
GET /v3/football/expected/fixtures
GET /v3/football/expected/lineups
```

### Key Includes for MTFI

```
# For Manager DNA
fixtures?include=formations,statistics,scores,participants

# For Squad Profiling  
players?include=statistics.details.type,position,detailedPosition,metadata

# For Match Analysis
fixtures?include=lineups.details.type,events,statistics.type
```

---

## 5. Data Gaps & Recommendations

### Critical Gaps (require workarounds or future enterprise data)

1. **Pressing/PPDA metrics** - No direct data
   - *Workaround*: Calculate proxy using tackles + interceptions + fouls per opposition possession
   - *Future*: StatsBomb provides PPDA

2. **Build-up play patterns** - No passing sequences
   - *Workaround*: Use pass completion %, long ball %, progressive passes (if available)
   - *Future*: StatsBomb/Opta provide passing networks

3. **Spatial/positional data** - No heatmaps or zones
   - *Workaround*: Use formation data + position-specific stats
   - *Future*: Tracking data providers (SkillCorner, Second Spectrum)

4. **Transition metrics** - No counter-attack data
   - *Workaround*: Fast goals after turnovers (manual calculation from events)
   - *Future*: StatsBomb sequences

### Acceptable for Prototype

The following can be delivered with Sportsmonks data:
- Manager formation preferences (strong)
- Basic tactical profile (possession, defensive record, xG)
- Squad player statistics (comprehensive)
- Player-position fit analysis (strong)
- Basic recruitment gap identification (moderate)

---

## 6. Subscription Tier Considerations

Sportsmonks pricing tiers:
- **Free**: Danish Superliga, Scottish Premiership only
- **European (€39/mo)**: Major European leagues ✅
- **Worldwide (€129/mo)**: All leagues
- **Enterprise**: Custom, higher limits

For MTFI prototype (e.g., Thomas Frank → Spurs):
- Need Premier League data
- **Minimum: European plan required**

Rate limits: 3000 requests per entity per hour (enterprise)

---

## 7. Day 1 Actions

- [ ] Confirm API credentials are working
- [ ] Test basic endpoints (leagues, teams, fixtures)
- [ ] Pull sample fixture data with formations + statistics
- [ ] Identify exact statistic type IDs needed
- [ ] Document any unexpected gaps or limitations
