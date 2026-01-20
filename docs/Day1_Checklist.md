# Day 1 Checklist: API Exploration & Documentation Review

**Date**: _____________  
**Project**: Aegis Platform - MTFI Prototype  
**Manager/Club**: _______________ → _______________

---

## Morning: Setup & Documentation (09:00-12:00)

### Environment Setup
- [ ] API credentials received and tested
- [ ] Python environment set up (requests, pandas, jupyter)
- [ ] Project folder structure created
- [ ] Git repository initialized (optional)

### API Documentation Review
- [ ] Reviewed Sportsmonks v3 documentation structure
- [ ] Identified key endpoints for MTFI:
  - [ ] Fixtures (with formations, statistics)
  - [ ] Coaches (with teams, fixtures, statistics)
  - [ ] Players (with statistics, positions, metadata)
  - [ ] Teams (with squads, statistics)
  - [ ] Types (statistic type definitions)

### Data Requirements Mapping
- [ ] Reviewed `01_Data_Requirements_Mapping.md`
- [ ] Confirmed understanding of what's available vs proxy needed

---

## Afternoon: Hands-on Exploration (13:00-17:00)

### API Testing
- [ ] Run `01_api_explorer.py` script
- [ ] Verified connection with API token
- [ ] Confirmed Premier League data is accessible

### Manager Data Pull
- [ ] Searched for target manager (e.g., Thomas Frank)
- [ ] Retrieved manager ID: _______________
- [ ] Pulled manager fixtures with formations
- [ ] Documented formation distribution:
  - Formation 1: _______________ (___%)
  - Formation 2: _______________ (___%)
  - Formation 3: _______________ (___%)

### Club Data Pull
- [ ] Identified target club ID: _______________
- [ ] Pulled squad roster with positions
- [ ] Pulled recent fixtures with statistics
- [ ] Confirmed player statistics available:
  - [ ] Goals/assists
  - [ ] Pass completion
  - [ ] Tackles/interceptions
  - [ ] Key passes
  - [ ] Match ratings

### Data Gap Log

| Expected Data | Available? | Notes/Workaround |
|---------------|------------|------------------|
| Formation per match | | |
| Possession % | | |
| xG/xGA | | |
| Pressing metrics (PPDA) | | |
| Pass sequences | | |
| Player positions | | |
| Player season stats | | |
| Market values | | |

---

## End of Day Summary

### What's Working Well
1. 
2. 
3. 

### Gaps/Limitations Found
1. 
2. 
3. 

### Questions for Gary (if any)
1. 
2. 

### Day 2 Priorities
1. 
2. 
3. 

---

## Quick Reference: Key Sportsmonks Endpoints

```bash
# Test connection
curl "https://api.sportmonks.com/v3/football/leagues?api_token=YOUR_TOKEN"

# Get Premier League teams
curl "https://api.sportmonks.com/v3/football/leagues/8?api_token=YOUR_TOKEN&include=currentSeason,teams"

# Search for coach
curl "https://api.sportmonks.com/v3/football/coaches/search/Thomas%20Frank?api_token=YOUR_TOKEN"

# Get fixture with full details
curl "https://api.sportmonks.com/v3/football/fixtures/FIXTURE_ID?api_token=YOUR_TOKEN&include=formations,statistics,lineups,scores"

# Get player with statistics
curl "https://api.sportmonks.com/v3/football/players/PLAYER_ID?api_token=YOUR_TOKEN&include=statistics.details.type,position,metadata"
```

---

## Notes Space

_Use this area for freeform notes during exploration:_

