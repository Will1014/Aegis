# Aegis Prototype

**Manager Tactical Fit Intelligence - Data Science Prototype**

Client: Gary Woodcock (Aegis Football Advisory Group)  
Timeline: 2-3 weeks  
Data Source: Sportsmonks API v3

---

## About Aegis

**Aegis Football Advisory Group** operates two complementary lines:
- **Aegis Advisory** — Services: coach/squad fit, recruitment planning, costed windows, succession planning
- **Aegis Platform** — Software: MTFI dashboards, scenario modelling, squad planner, reporting

This prototype forms the foundation of the **Aegis Platform** product line.

---

## Project Structure

```
Aegis_Prototype/
├── data/
│   ├── raw/              # Raw API responses (JSON)
│   └── processed/        # Cleaned/transformed data
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Python scripts
│   └── 01_api_explorer.py
├── reports/              # Technical documentation
├── dashboards/           # Power BI files
├── docs/                 # Working documents
│   ├── 01_Data_Requirements_Mapping.md
│   └── Day1_Checklist.md
└── requirements.txt
```

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API token**
   - Edit `scripts/01_api_explorer.py`
   - Replace `YOUR_API_TOKEN_HERE` with your Sportsmonks token

3. **Run API explorer**
   ```bash
   cd scripts
   python 01_api_explorer.py
   ```

4. **Check outputs**
   - Raw JSON files saved to `data/raw/`

---

## Key Deliverables

1. **Data Pipeline Diagram** - Architecture showing data flow
2. **Technical Report** - Methodology, findings, limitations
3. **Code Repository** - This folder with notebooks/scripts
4. **Power BI Dashboard** - Manager DNA, Squad Fit, Recruitment

---

## Weekly Milestones

| Week | Focus | Outputs |
|------|-------|---------|
| 1 | Data Discovery & Pipeline | API exploration, ETL pipeline, data quality report |
| 2 | Manager DNA & Squad Fit | Manager profile model, fit scoring, player classifications |
| 3 | Visualisation & Delivery | Power BI dashboard, final report, handover |

---

## Data Sources

**Primary**: Sportsmonks Football API v3
- Fixtures, formations, match statistics
- Player statistics and profiles
- Coach/manager data
- xG metrics

**Limitations documented in**: `docs/01_Data_Requirements_Mapping.md`

---

## Contact

Consultant: [Your Name]  
Client: Gary Woodcock (gary.woodcock@live.co.uk)  
Organisation: Aegis Football Advisory Group
