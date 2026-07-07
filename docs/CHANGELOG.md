# Changelog

All notable changes to Aegis MTFI are documented here.

---

## [1.0] - June 2026

First official release of the MTFI platform.

### Added

- Five-tab interface: Report, Formation, DNA Profile, Squad, Dashboard
- Automated briefing reports - five sections, auto-generated on every analysis run, downloadable as PDF or HTML
- Manager Shortlist Ranker - rank up to ten managers against a single squad simultaneously
- Formation Analysis - live StatsBomb lineup data, compatibility scoring, dual pitch diagrams, formation history charts, player transition risk table
- Manager DNA Insights panel - tactically similar managers, pillar vs archetype benchmark, confidence indicator
- Player Dossier mode - standalone scouting reports with percentile rankings and downloadable HTML
- Master pre-trained model - trained nightly across all five licensed leagues and four seasons via GitHub Actions
- Cross-league analysis - any manager against any club regardless of competition
- Full squad table with detailed positions, classification filter, and Ideal XI slot column
- Same-club comparison delta table showing fit score change per player under each manager
- Six tactical archetypes replacing the previous two-cluster output
- Automatic league detection from selected club - league selector removed from inputs

### Changed

- Dashboard moved to a dedicated tab rather than the primary view
- Manager pool now draws from all four available seasons, not just the current season
- Formation values no longer silently default to 4-3-3 - unavailable data is surfaced explicitly

### Fixed

- Formation bar chart previously showed hardcoded placeholder data - now uses live StatsBomb lineup records
- Pitch player positions were hardcoded to 4-3-3 regardless of manager - now dynamically positioned per formation
- Ideal XI slot system was blind to the manager's actual shape - now formation-aware across seven supported formations
