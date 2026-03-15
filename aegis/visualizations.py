"""
Aegis Visualizations
====================
Generate charts and dashboards for analysis results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .config import Config


class AegisVisualizer:
    """
    Generate visualizations for Aegis analysis.
    
    Usage:
        from aegis import AegisVisualizer
        
        viz = AegisVisualizer()
        viz.load_results()
        viz.plot_all()
        
        # Or individual charts
        viz.plot_dna_radar()
        viz.plot_squad_fit()
        viz.plot_ideal_xi()
        
        # Generate interactive HTML dashboard
        viz.generate_dashboard()
    """
    
    # Colours
    COLORS = {
        "Key Enabler": "#22c55e",      # Green
        "Good Fit": "#eab308",          # Yellow
        "System Dependent": "#f97316",  # Orange
        "Potentially Marginalised": "#ef4444",  # Red
        "primary": "#3b82f6",           # Blue
        "secondary": "#64748b"          # Slate
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory containing analysis results.
                       Defaults to Config.OUTPUT_DIR
        """
        self.output_dir = Path(output_dir) if output_dir else Config.OUTPUT_DIR
        self.results = None
        self.squad_fit_data = None
        
        # Check for matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            self.plt = plt
            self.mpatches = mpatches
        except ImportError:
            print("⚠ matplotlib not installed. Run: pip install matplotlib")
            self.plt = None
    
    def load_results(self):
        """Load analysis results from JSON."""
        results_file = self.output_dir / "aegis_analysis.json"
        if results_file.exists():
            with open(results_file) as f:
                self.results = json.load(f)
            print(f"✓ Loaded results for: {self.results.get('manager', 'Unknown')}")
        else:
            raise FileNotFoundError(f"Results not found: {results_file}")
        
        # Also load squad fit CSV for dashboard
        squad_file = self.output_dir / "squad_fit_scores.csv"
        if squad_file.exists():
            with open(squad_file) as f:
                reader = csv.DictReader(f)
                self.squad_fit_data = list(reader)
        
        return self
    
    def plot_all(self):
        """Generate all visualizations."""
        if not self.plt:
            print("matplotlib required for visualizations")
            return
        
        print("\nGenerating visualizations...")
        
        self.plot_dna_radar()
        self.plot_formation_usage()
        self.plot_squad_fit()
        self.plot_classification_pie()
        self.plot_ideal_xi()
        self.plot_recruitment()
        self.plot_executive_summary()
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
    
    # =========================================================================
    # INTERACTIVE HTML DASHBOARD
    # =========================================================================
    
    def generate_dashboard(
        self,
        filename: str = "MTFI_Dashboard.html",
        manager_name: Optional[str] = None,
        target_club: Optional[str] = None,
        season: str = "2024/25"
    ):
        """
        Generate MTFI Dashboard v2 with clustering visualization.
        
        This is the advanced dashboard with:
        - Manager DNA radar chart
        - PCA cluster scatter plot
        - Ideal XI pitch visualization
        - Positional fit analysis bars
        - Recruitment priorities table
        - Full squad fit scores
        
        Args:
            filename: Output filename (default: MTFI_Dashboard.html)
            manager_name: Override manager name (auto-detected if None)
            target_club: Override target club (auto-detected if None)
            season: Season label for display
        
        Returns:
            Path to generated dashboard file
        """
        print("\n" + "=" * 50)
        print("GENERATING MTFI DASHBOARD v2")
        print("=" * 50)
        
        # Load data files
        print("\n[1/3] Loading data files...")
        
        manager_profiles = self._load_csv("manager_profiles.csv")
        cluster_centroids = self._load_csv("cluster_centroids.csv")
        squad_fit = self._load_csv("squad_fit_scores.csv")
        ideal_xi = self._load_csv("ideal_xi.csv")
        recruitment = self._load_csv("recruitment_priorities.csv")
        positional_gaps = self._load_csv("positional_gaps.csv")
        
        # Try to load summary JSON
        summary = {}
        summary_path = self.output_dir / "squad_fit_summary.json"
        aegis_summary_path = self.output_dir / "aegis_analysis.json"
        
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  ✓ Loaded: squad_fit_summary.json")
        elif aegis_summary_path.exists():
            with open(aegis_summary_path) as f:
                summary = json.load(f)
            print(f"  ✓ Loaded: aegis_analysis.json")
        
        # Determine manager and club
        print("\n[2/3] Processing data...")
        
        if not manager_name:
            manager_name = summary.get("target_manager") or summary.get("manager") or "Unknown Manager"
        
        if not target_club:
            target_club = summary.get("target_club") or summary.get("club") or "Target Club"
        
        print(f"  Manager: {manager_name}")
        print(f"  Club: {target_club}")
        
        # Extract 8-pillar DNA dimensions from summary JSON
        dna_dimensions = summary.get("dna_dimensions", {})
        if dna_dimensions:
            print(f"  ✓ Loaded 8-pillar DNA dimensions ({len(dna_dimensions)} pillars)")
        else:
            print(f"  ⚠ No dna_dimensions in summary JSON — radar will use defaults")
        
        # Generate HTML
        print("\n[3/3] Generating HTML...")
        
        html = self._generate_dashboard_v2_html(
            manager_profiles=manager_profiles,
            cluster_centroids=cluster_centroids,
            squad_fit=squad_fit,
            ideal_xi=ideal_xi,
            recruitment=recruitment,
            positional_gaps=positional_gaps,
            manager_name=manager_name,
            target_club=target_club,
            season=season,
            dna_dimensions=dna_dimensions
        )
        
        # Save
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"\n✓ Dashboard saved: {output_path}")
        print("=" * 50)
        
        return output_path
    
    def _load_csv(self, filename: str) -> List[Dict]:
        """Load CSV file from output directory."""
        filepath = self.output_dir / filename
        
        # Also check training directory for manager profiles
        if not filepath.exists() and filename in ["manager_profiles.csv", "cluster_centroids.csv"]:
            training_dir = self.output_dir.parent / "processed" / "training"
            filepath = training_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ Not found: {filename}")
            return []
        
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print(f"  ✓ Loaded: {filename} ({len(data)} rows)")
        return data
    
    def _safe_float(self, val, default=0.0):
        """Safely convert to float."""
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, val, default=0):
        """Safely convert to int."""
        try:
            return int(float(val)) if val else default
        except (ValueError, TypeError):
            return default
    
    def _generate_dashboard_v2_html(
        self,
        manager_profiles: List[Dict],
        cluster_centroids: List[Dict],
        squad_fit: List[Dict],
        ideal_xi: List[Dict],
        recruitment: List[Dict],
        positional_gaps: List[Dict],
        manager_name: str,
        target_club: str,
        season: str,
        dna_dimensions: Optional[Dict] = None
    ) -> str:
        """Generate the MTFI Dashboard v2 HTML."""
        from datetime import datetime
        
        # Find target manager in profiles
        target_manager_data = None
        for m in manager_profiles:
            if manager_name.lower() in m.get("coach_name", "").lower():
                target_manager_data = m
                break
        
        if not target_manager_data and manager_profiles:
            target_manager_data = manager_profiles[0]
        
        # Process clusters
        clusters_js = []
        cluster_colors = ['#3B82F6', '#EF4444', '#22C55E', '#8B5CF6', '#F59E0B', '#EC4899', '#14B8A6']
        for i, c in enumerate(cluster_centroids):
            clusters_js.append({
                "id": self._safe_int(c.get("cluster")),
                "name": c.get("cluster_name", f"Cluster {c.get('cluster')}"),
                "color": cluster_colors[i % len(cluster_colors)]
            })
        
        # Process managers for scatter
        managers_js = []
        for m in manager_profiles:
            managers_js.append({
                "name": m.get("coach_name", "Unknown"),
                "team": m.get("team_name", "Unknown"),
                "pca1": self._safe_float(m.get("pca_1")),
                "pca2": self._safe_float(m.get("pca_2")),
                "cluster": self._safe_int(m.get("cluster"))
            })
        
        # Process squad fit
        squad_js = []
        for p in squad_fit:
            squad_js.append({
                "name": p.get("name") or p.get("Name", "Unknown"),
                "position": p.get("position") or p.get("Position", "Unknown"),
                "group": p.get("position_group", "MID"),
                "fit": self._safe_float(p.get("fit_score") or p.get("Fit Score")),
                "classification": p.get("classification") or p.get("Classification", "Unknown")
            })
        
        # Process ideal XI
        pitch_positions = {
            "GK": {"x": 50, "y": 90}, "LB": {"x": 15, "y": 68},
            "CB1": {"x": 35, "y": 73}, "CB": {"x": 35, "y": 73}, "CB2": {"x": 65, "y": 73},
            "RB": {"x": 85, "y": 68}, "DM": {"x": 50, "y": 52},
            "CM": {"x": 30, "y": 40}, "AM": {"x": 70, "y": 40},
            "LW": {"x": 15, "y": 22}, "CF": {"x": 50, "y": 15}, "RW": {"x": 85, "y": 22}
        }
        
        ideal_js = []
        cb_count = 0
        for p in ideal_xi:
            slot = p.get("slot") or p.get("position", "")
            if slot == "CB":
                cb_count += 1
                pos_key = "CB1" if cb_count == 1 else "CB2"
            else:
                pos_key = slot
            pos = pitch_positions.get(pos_key, {"x": 50, "y": 50})
            ideal_js.append({
                "slot": slot,
                "name": p.get("name", "Unknown"),
                "fit": self._safe_float(p.get("fit_score")),
                "x": pos["x"], "y": pos["y"]
            })
        
        # Process positional gaps (or generate from squad)
        gaps_js = []
        if positional_gaps:
            for g in positional_gaps:
                gaps_js.append({
                    "position": g.get("position", "Unknown"),
                    "avgFit": self._safe_float(g.get("avg_fit") or g.get("avgFit")),
                    "maxFit": self._safe_float(g.get("max_fit") or g.get("maxFit")),
                    "count": self._safe_int(g.get("count")),
                    "gap": self._safe_float(g.get("gap"))
                })
        elif squad_js:
            # Generate from squad data
            position_stats = {}
            for p in squad_js:
                pos = p["position"]
                if pos not in position_stats:
                    position_stats[pos] = {"fits": [], "max": 0}
                position_stats[pos]["fits"].append(p["fit"])
                position_stats[pos]["max"] = max(position_stats[pos]["max"], p["fit"])
            
            for pos, stats in position_stats.items():
                avg = sum(stats["fits"]) / len(stats["fits"])
                gaps_js.append({
                    "position": pos,
                    "avgFit": round(avg, 1),
                    "maxFit": stats["max"],
                    "count": len(stats["fits"]),
                    "gap": round(max(0, 75 - avg), 1)
                })
            gaps_js.sort(key=lambda x: x["avgFit"], reverse=True)
        
        # Process recruitment
        recruit_js = []
        for i, r in enumerate(recruitment):
            recruit_js.append({
                "rank": i + 1,
                "position": r.get("position") or r.get("Position", "Unknown"),
                "urgency": r.get("urgency") or r.get("Urgency", "Medium"),
                "timeline": r.get("timeline") or r.get("Timeline", "Summer"),
                "gap": self._safe_float(r.get("gap") or r.get("Gap")),
                "costLow": self._safe_int(r.get("cost_low") or r.get("Cost Low")),
                "costHigh": self._safe_int(r.get("cost_high") or r.get("Cost High"))
            })
        
        # Build manager data
        manager_js = {
            "name": manager_name,
            "team": target_club,
            "cluster": self._safe_int(target_manager_data.get("cluster")) if target_manager_data else 0,
            "clusterName": target_manager_data.get("cluster_name", "Unknown") if target_manager_data else "Unknown",
            "goalsScored": self._safe_float(target_manager_data.get("goals_scored", 1.5)) if target_manager_data else 1.5,
            "goalsConceded": self._safe_float(target_manager_data.get("goals_conceded", 1.5)) if target_manager_data else 1.5,
            "cleanSheetPct": self._safe_float(target_manager_data.get("_clean_sheet_pct") or target_manager_data.get("clean_sheet_pct", 25)) if target_manager_data else 25,
            "winRate": self._safe_float(target_manager_data.get("_win_rate") or target_manager_data.get("win_rate", 33)) if target_manager_data else 33,
            "possession": self._safe_float(target_manager_data.get("_possession") or target_manager_data.get("possession", 50)) if target_manager_data else 50,
            "passAccuracy": self._safe_float(target_manager_data.get("_pass_accuracy") or target_manager_data.get("pass_accuracy", 80)) if target_manager_data else 80,
            "shots": self._safe_float(target_manager_data.get("_np_shots_pg") or target_manager_data.get("shots", 12)) if target_manager_data else 12,
            "tackles": self._safe_float(target_manager_data.get("_pressures_pg") or target_manager_data.get("tackles", 15)) if target_manager_data else 15,
            "interceptions": self._safe_float(target_manager_data.get("_deep_completions_pg") or target_manager_data.get("interceptions", 10)) if target_manager_data else 10
        }
        
        # Build 8-pillar DNA dimensions for radar chart
        # Priority: explicit dna_dimensions param > aegis_analysis.json > defaults
        if not dna_dimensions and target_manager_data:
            # Derive from StatsBomb features in manager_profiles.csv
            press = self._safe_float(target_manager_data.get("pressing_intensity", 50))
            patience = self._safe_float(target_manager_data.get("build_up_patience", 50))
            chance_q = self._safe_float(target_manager_data.get("chance_quality", 0.1))
            def_height = self._safe_float(target_manager_data.get("defensive_line_height", 40))
            width = self._safe_float(target_manager_data.get("width_usage", 5))
            sp_emph = self._safe_float(target_manager_data.get("set_piece_emphasis", 20))
            trans = self._safe_float(target_manager_data.get("transition_threat", 1))
            def_solid = self._safe_float(target_manager_data.get("defensive_solidity", 50))
            counterpress = self._safe_float(target_manager_data.get("counterpress_rate", 30))
            possession_val = self._safe_float(target_manager_data.get("_possession", 50))
            
            dna_dimensions = {
                "Shape & Occupation": min(100, round(def_height * 1.2 + possession_val * 0.3, 0)),
                "Build-up": min(100, round(patience, 0)),
                "Chance Creation": min(100, round(chance_q * 400, 0)),
                "Press & Counterpress": min(100, round((press + counterpress) / 2, 0)),
                "Block & Line Height": min(100, round(def_height + def_solid * 0.3, 0)),
                "Transitions": min(100, round(trans * 25, 0)),
                "Width & Overloads": min(100, round(width * 10, 0)),
                "Set Pieces": min(100, round(sp_emph * 2, 0)),
            }
        
        if not dna_dimensions:
            dna_dimensions = {
                "Shape & Occupation": 50, "Build-up": 50, "Chance Creation": 50,
                "Press & Counterpress": 50, "Block & Line Height": 50,
                "Transitions": 50, "Width & Overloads": 50, "Set Pieces": 50,
            }
        
        # JSON encode
        manager_json = json.dumps(manager_js)
        dna_dimensions_json = json.dumps(dna_dimensions)
        clusters_json = json.dumps(clusters_js)
        managers_json = json.dumps(managers_js)
        squad_json = json.dumps(squad_js)
        ideal_json = json.dumps(ideal_js)
        gaps_json = json.dumps(gaps_js)
        recruit_json = json.dumps(recruit_js)
        
        # Generate the HTML (inline React dashboard)
        html = self._get_dashboard_v2_template(
            manager_json=manager_json,
            dna_dimensions_json=dna_dimensions_json,
            clusters_json=clusters_json,
            managers_json=managers_json,
            squad_json=squad_json,
            ideal_json=ideal_json,
            gaps_json=gaps_json,
            recruit_json=recruit_json,
            season=season,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        
        return html
    
    def _get_dashboard_v2_template(
        self,
        manager_json: str,
        dna_dimensions_json: str,
        clusters_json: str,
        managers_json: str,
        squad_json: str,
        ideal_json: str,
        gaps_json: str,
        recruit_json: str,
        season: str,
        generated_at: str
    ) -> str:
        """Return the full HTML template for Dashboard v2."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MTFI Dashboard v2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>body {{ margin: 0; }} * {{ box-sizing: border-box; }}</style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
const {{ useState }} = React;

const MTFIDashboard = () => {{
  const managerData = {manager_json};
  const dnaPillars = {dna_dimensions_json};
  const clusters = {clusters_json};
  const allManagers = {managers_json};
  const squadFit = {squad_json};
  const idealXI = {ideal_json};
  const positionGaps = {gaps_json};
  const recruitment = {recruit_json};
  const season = "{season}";

  const totalPlayers = squadFit.length;
  const keyEnablers = squadFit.filter(p => p.fit >= 75).length;
  const goodFit = squadFit.filter(p => p.fit >= 60 && p.fit < 75).length;
  const systemDependent = squadFit.filter(p => p.fit >= 45 && p.fit < 60).length;
  const marginalised = squadFit.filter(p => p.fit < 45).length;
  const avgFit = totalPlayers > 0 ? (squadFit.reduce((a, b) => a + b.fit, 0) / totalPlayers).toFixed(1) : 0;
  const investmentLow = recruitment.reduce((a, b) => a + (b.costLow || 0), 0);
  const investmentHigh = recruitment.reduce((a, b) => a + (b.costHigh || 0), 0);

  const getFitColor = (score) => {{
    if (score >= 75) return '#22C55E';
    if (score >= 60) return '#84CC16';
    if (score >= 45) return '#EAB308';
    if (score >= 30) return '#F97316';
    return '#EF4444';
  }};

  const getClassificationColor = (cls) => ({{
    'Key Enabler': '#22C55E', 'Good Fit': '#84CC16',
    'System Dependent': '#EAB308', 'Potentially Marginalised': '#EF4444'
  }})[cls] || '#64748B';

  const KPICard = ({{ title, value, subtitle, color, icon }}) => (
    <div className="bg-slate-800/60 backdrop-blur border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-slate-400 text-xs uppercase tracking-wider font-medium">{{title}}</div>
          <div className={{`text-2xl font-bold mt-1 ${{color || 'text-white'}}`}}>{{value}}</div>
          {{subtitle && <div className="text-slate-500 text-sm mt-1">{{subtitle}}</div>}}
        </div>
        {{icon && <div className="text-2xl opacity-50">{{icon}}</div>}}
      </div>
    </div>
  );

  const ClusterScatter = () => {{
    if (allManagers.length === 0) return <div className="text-slate-500 text-center py-8">No manager data</div>;
    const minX = Math.min(...allManagers.map(m => m.pca1)) - 0.5;
    const maxX = Math.max(...allManagers.map(m => m.pca1)) + 0.5;
    const minY = Math.min(...allManagers.map(m => m.pca2)) - 0.5;
    const maxY = Math.max(...allManagers.map(m => m.pca2)) + 0.5;
    const scaleX = (val) => ((val - minX) / (maxX - minX)) * 100;
    const scaleY = (val) => 100 - ((val - minY) / (maxY - minY)) * 100;
    const clusterColors = {{}};
    clusters.forEach(c => {{ clusterColors[c.id] = c.color; }});
    return (
      <div className="flex flex-col h-full">
        <div className="relative flex-1 bg-slate-900/50 rounded-xl min-h-[220px]">
          {{allManagers.map((m, i) => (
            <div key={{i}}
              className={{`absolute w-3.5 h-3.5 rounded-full transform -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-all hover:scale-150 hover:z-20 ${{m.name === managerData.name ? 'ring-2 ring-white ring-offset-1 ring-offset-slate-900 z-10 w-5 h-5' : ''}}`}}
              style={{{{ left: `${{2 + scaleX(m.pca1) * 0.96}}%`, top: `${{2 + scaleY(m.pca2) * 0.96}}%`, backgroundColor: clusterColors[m.cluster] || '#64748B' }}}}
              title={{`${{m.name}} (${{m.team}})`}}>
              {{m.name === managerData.name && (
                <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-xs font-bold text-white whitespace-nowrap bg-slate-800/90 px-1.5 py-0.5 rounded">{{managerData.name}}</div>
              )}}
            </div>
          ))}}
        </div>
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-3 pt-3 border-t border-slate-700/50">
          {{clusters.map((c) => (
            <div key={{c.id}} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{{{ backgroundColor: c.color }}}} />
              <span className="text-slate-400 text-xs">{{c.name}}</span>
            </div>
          ))}}
        </div>
      </div>
    );
  }};

  const FormationPitch = () => (
    <div className="relative w-full aspect-[3/4] bg-gradient-to-b from-emerald-900/90 to-emerald-800/90 rounded-xl overflow-hidden border border-emerald-700/30">
      <div className="absolute inset-3 border-2 border-white/20 rounded">
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-white/20" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-14 h-14 border-2 border-white/20 rounded-full" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-28 h-10 border-2 border-t-0 border-white/20" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-28 h-10 border-2 border-b-0 border-white/20" />
      </div>
      {{idealXI.map((player, i) => (
        <div key={{i}} className="absolute transform -translate-x-1/2 -translate-y-1/2 text-center group" style={{{{ left: `${{player.x}}%`, top: `${{player.y}}%` }}}}>
          <div className="w-9 h-9 rounded-full flex items-center justify-center text-white font-bold text-xs border-2 shadow-lg transition-transform group-hover:scale-110"
            style={{{{ backgroundColor: getFitColor(player.fit), borderColor: 'rgba(255,255,255,0.3)' }}}}>
            {{Math.round(player.fit)}}
          </div>
          <div className="text-white text-xs mt-1 font-medium drop-shadow-lg opacity-90">{{player.name.split(' ').pop()}}</div>
          <div className="text-white/60 text-xs">{{player.slot}}</div>
        </div>
      ))}}
    </div>
  );

  const PositionFitBars = () => (
    <div className="space-y-2">
      {{positionGaps.map((pos, i) => (
        <div key={{i}} className="flex items-center gap-3">
          <div className="w-28 text-sm text-slate-300 truncate">{{pos.position}}</div>
          <div className="flex-1 h-5 bg-slate-700/50 rounded-full overflow-hidden relative">
            <div className="h-full rounded-full" style={{{{ width: `${{Math.min(pos.avgFit, 100)}}%`, backgroundColor: getFitColor(pos.avgFit) }}}} />
            {{pos.maxFit > 0 && <div className="absolute top-0 h-full w-0.5 bg-white/60" style={{{{ left: `${{Math.min(pos.maxFit, 100)}}%` }}}} />}}
            <div className="absolute top-0 h-full w-px bg-white/30 left-3/4" />
          </div>
          <div className="w-10 text-right font-mono text-sm text-slate-300">{{pos.avgFit.toFixed(1)}}</div>
          <div className="w-6 text-center text-slate-500 text-xs">({{pos.count}})</div>
        </div>
      ))}}
    </div>
  );

  const SquadTable = () => (
    <div className="overflow-auto max-h-80">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-slate-800">
          <tr className="text-slate-400 text-xs uppercase tracking-wider">
            <th className="px-3 py-2 text-left">Player</th>
            <th className="px-3 py-2 text-left">Position</th>
            <th className="px-3 py-2 text-center">Fit</th>
            <th className="px-3 py-2 text-left">Classification</th>
          </tr>
        </thead>
        <tbody>
          {{[...squadFit].sort((a, b) => b.fit - a.fit).map((player, i) => (
            <tr key={{i}} className="border-t border-slate-700/30 hover:bg-slate-700/20">
              <td className="px-3 py-2 font-medium">{{player.name}}</td>
              <td className="px-3 py-2 text-slate-400">{{player.position}}</td>
              <td className="px-3 py-2 text-center">
                <span className="inline-block w-10 py-0.5 rounded text-xs font-bold" style={{{{ backgroundColor: getFitColor(player.fit) + '30', color: getFitColor(player.fit) }}}}>{{player.fit.toFixed(0)}}</span>
              </td>
              <td className="px-3 py-2">
                <span className="text-xs px-2 py-0.5 rounded" style={{{{ backgroundColor: getClassificationColor(player.classification) + '20', color: getClassificationColor(player.classification) }}}}>{{player.classification}}</span>
              </td>
            </tr>
          ))}}
        </tbody>
      </table>
    </div>
  );

  const RecruitmentTable = () => (
    <div>
      <table className="w-full text-sm">
        <thead className="bg-slate-700/30">
          <tr className="text-slate-400 text-xs uppercase tracking-wider">
            <th className="px-4 py-3 text-left">#</th>
            <th className="px-4 py-3 text-left">Position</th>
            <th className="px-4 py-3 text-center">Urgency</th>
            <th className="px-4 py-3 text-center">Gap</th>
            <th className="px-4 py-3 text-center">Window</th>
            <th className="px-4 py-3 text-right">Est. Cost</th>
          </tr>
        </thead>
        <tbody>
          {{recruitment.map((r, i) => (
            <tr key={{i}} className="border-t border-slate-700/30 hover:bg-slate-700/20">
              <td className="px-4 py-3 font-bold text-slate-500">{{r.rank}}</td>
              <td className="px-4 py-3 font-medium">{{r.position}}</td>
              <td className="px-4 py-3 text-center">
                <span className={{`px-2 py-1 rounded text-xs font-medium ${{r.urgency === 'Critical' ? 'bg-red-500/20 text-red-400' : r.urgency === 'High' ? 'bg-orange-500/20 text-orange-400' : 'bg-amber-500/20 text-amber-400'}}`}}>{{r.urgency}}</span>
              </td>
              <td className="px-4 py-3 text-center font-mono">{{r.gap.toFixed(1)}}</td>
              <td className="px-4 py-3 text-center text-slate-400">{{r.timeline}}</td>
              <td className="px-4 py-3 text-right font-mono">£{{r.costLow}}-{{r.costHigh}}M</td>
            </tr>
          ))}}
        </tbody>
        {{recruitment.length > 0 && (
          <tfoot className="bg-slate-700/20 border-t border-slate-600/50">
            <tr>
              <td colSpan={{5}} className="px-4 py-3 text-right font-medium text-slate-300">Total Investment</td>
              <td className="px-4 py-3 text-right font-bold text-lg text-white">£{{investmentLow}}-{{investmentHigh}}M</td>
            </tr>
          </tfoot>
        )}}
      </table>
    </div>
  );

  const ManagerDNARadar = () => {{
    const dims = Object.entries(dnaPillars);
    const n = dims.length;
    const angleStep = (2 * Math.PI) / n;
    const centerX = 120, centerY = 120, maxRadius = 90;
    const points = dims.map(([label, value], i) => {{
      const angle = i * angleStep - Math.PI / 2;
      const radius = (Math.min(value, 100) / 100) * maxRadius;
      return {{ x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius, label, value }};
    }});
    const polygonPoints = points.map(p => `${{p.x}},${{p.y}}`).join(' ');
    return (
      <div className="flex flex-col items-center">
        <svg width="280" height="280" className="overflow-visible">
          {{[25, 50, 75, 100].map((level) => (<circle key={{level}} cx={{centerX}} cy={{centerY}} r={{(level / 100) * maxRadius}} fill="none" stroke="#334155" strokeWidth="1" opacity="0.5" />))}}
          {{dims.map((_, i) => {{ const angle = i * angleStep - Math.PI / 2; return (<line key={{i}} x1={{centerX}} y1={{centerY}} x2={{centerX + Math.cos(angle) * maxRadius}} y2={{centerY + Math.sin(angle) * maxRadius}} stroke="#334155" strokeWidth="1" opacity="0.5" />); }})}}
          <polygon points={{polygonPoints}} fill="#F59E0B" fillOpacity="0.3" stroke="#F59E0B" strokeWidth="2" />
          {{points.map((p, i) => (<circle key={{i}} cx={{p.x}} cy={{p.y}} r="4" fill="#F59E0B" stroke="#FCD34D" strokeWidth="2" />))}}
          {{dims.map(([label], i) => {{ const angle = i * angleStep - Math.PI / 2; const labelRadius = maxRadius + 22; const x = centerX + Math.cos(angle) * labelRadius; const y = centerY + Math.sin(angle) * labelRadius; return (<text key={{i}} x={{x}} y={{y}} textAnchor="middle" dominantBaseline="middle" className="fill-slate-400" style={{{{fontSize: '9px'}}}}>{{label}}</text>); }})}}
          {{points.map((p, i) => (<text key={{'v'+i}} x={{p.x}} y={{p.y - 10}} textAnchor="middle" className="fill-amber-300" style={{{{fontSize: '10px', fontWeight: 'bold'}}}}>{{Math.round(p.value)}}</text>))}}
        </svg>
        <div className="mt-4 px-4 py-2 rounded-full bg-amber-500/20 border border-amber-500/50">
          <span className="text-amber-400 font-semibold">Cluster: {{managerData.clusterName}}</span>
        </div>
      </div>
    );
  }};

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center font-black text-lg">A</div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">Manager Tactical Fit Intelligence</h1>
                <p className="text-slate-400 text-sm">Aegis Football Advisory Group</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-semibold">{{managerData.name}} → {{managerData.team}}</div>
              <div className="text-slate-400 text-sm">{{season}} Season Analysis</div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          <KPICard title="Squad Avg Fit" value={{`${{avgFit}}%`}} subtitle={{`${{totalPlayers}} players`}} icon="📊" />
          <KPICard title="Key Enablers" value={{keyEnablers}} subtitle="≥75" color="text-emerald-400" icon="🟢" />
          <KPICard title="Good Fit" value={{goodFit}} subtitle="60-74" color="text-lime-400" icon="🟡" />
          <KPICard title="System Dependent" value={{systemDependent}} subtitle="45-59" color="text-amber-400" icon="🟠" />
          <KPICard title="Marginalised" value={{marginalised}} subtitle="<45" color="text-red-400" icon="🔴" />
          <KPICard title="Investment Req." value={{`£${{investmentLow}}-${{investmentHigh}}M`}} subtitle={{`${{recruitment.length}} priorities`}} icon="💰" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5">
            <h2 className="text-lg font-semibold mb-2 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-amber-500"></span>Manager DNA</h2>
            <div className="text-center mb-2">
              <div className="text-xl font-bold">{{managerData.name}}</div>
              <div className="text-slate-400 text-sm">{{managerData.team}}</div>
            </div>
            <ManagerDNARadar />
            <div className="grid grid-cols-2 gap-2 mt-4 text-sm">
              <div className="bg-slate-700/30 rounded-lg p-2 text-center">
                <div className="text-slate-400 text-xs">Goals/Game</div>
                <div className="font-bold text-lg">{{managerData.goalsScored?.toFixed(2) || 'N/A'}}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-2 text-center">
                <div className="text-slate-400 text-xs">Conceded/Game</div>
                <div className="font-bold text-lg">{{managerData.goalsConceded?.toFixed(2) || 'N/A'}}</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5 flex flex-col">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-violet-500"></span>Manager Clustering (PCA)</h2>
            <ClusterScatter />
            <div className="mt-3 text-sm">
              <span className="text-amber-400 font-semibold">{{managerData.name}}</span>
              <span className="text-slate-400"> clusters with </span>
              <span className="text-white font-semibold">{{managerData.clusterName}}</span>
              <span className="text-slate-400"> managers</span>
            </div>
          </div>

          <div className="bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span>Ideal XI (4-3-3)</h2>
            <FormationPitch />
            <div className="mt-3 flex justify-center gap-3 text-xs flex-wrap">
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-emerald-500"></div><span className="text-slate-400">≥75</span></div>
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-lime-500"></div><span className="text-slate-400">60-74</span></div>
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-yellow-500"></div><span className="text-slate-400">45-59</span></div>
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-red-500"></div><span className="text-slate-400">&lt;45</span></div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500"></span>Positional Fit Analysis</h2>
            {{positionGaps.length > 0 ? <PositionFitBars /> : <div className="text-slate-500 text-center py-4">No positional data</div>}}
          </div>

          <div className="bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-500"></span>Recruitment Priorities</h2>
            {{recruitment.length > 0 ? <RecruitmentTable /> : <div className="text-slate-500 text-center py-4">No recruitment priorities</div>}}
          </div>
        </div>

        <div className="mt-6 bg-slate-800/40 backdrop-blur border border-slate-700/50 rounded-2xl p-5">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-cyan-500"></span>Full Squad Fit Scores</h2>
          {{squadFit.length > 0 ? <SquadTable /> : <div className="text-slate-500 text-center py-4">No squad data</div>}}
        </div>

        <div className="mt-8 pt-6 border-t border-slate-800 flex items-center justify-between text-sm text-slate-500">
          <div>MTFI Prototype v0.2 • Data Source: StatsBomb</div>
          <div>Generated: {generated_at}</div>
        </div>
      </div>
    </div>
  );
}};

ReactDOM.render(<MTFIDashboard />, document.getElementById('root'));
    </script>
</body>
</html>'''
    
    def _generate_dashboard_html(
        self,
        manager: str,
        primary_formation: str,
        matches: int,
        dna_dimensions: Dict,
        squad_summary: Dict,
        squad_fit: List,
        ideal_xi: List,
        recruitment: List
    ) -> str:
        """Generate the HTML content for the dashboard."""
        
        # Convert data to JSON for embedding
        dna_json = json.dumps(dna_dimensions)
        squad_fit_json = json.dumps(squad_fit)
        ideal_xi_json = json.dumps(ideal_xi)
        recruitment_json = json.dumps(recruitment)
        squad_summary_json = json.dumps(squad_summary)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS MTFI Dashboard - {manager}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        .tab-active {{ border-bottom: 3px solid #3b82f6; color: #1e40af; font-weight: 600; }}
    </style>
</head>
<body class="bg-gray-100">
    <div id="root"></div>
    
    <script type="text/babel">
        const {{ useState }} = React;

        // ============ EMBEDDED DATA ============
        const managerName = "{manager}";
        const primaryFormation = "{primary_formation}";
        const matchesAnalysed = {matches};
        const dnaDimensions = {dna_json};
        const squadSummary = {squad_summary_json};
        const squadFit = {squad_fit_json};
        const idealXI = {ideal_xi_json};
        const recruitment = {recruitment_json};

        // ============ COMPONENTS ============
        
        const classificationColors = {{
            "Key Enabler": "#22c55e",
            "Good Fit": "#eab308",
            "System Dependent": "#f97316",
            "Potentially Marginalised": "#ef4444"
        }};

        const urgencyColors = {{
            "Critical": "#ef4444",
            "High": "#f97316",
            "Medium": "#eab308"
        }};

        // DNA Radar Chart (SVG)
        const DNARadar = () => {{
            const dims = Object.entries(dnaDimensions);
            const count = dims.length;
            const angleStep = (2 * Math.PI) / count;
            const cx = 150, cy = 150, radius = 100;
            
            const points = dims.map(([label, value], i) => {{
                const angle = i * angleStep - Math.PI / 2;
                const r = (value / 100) * radius;
                return {{
                    x: cx + r * Math.cos(angle),
                    y: cy + r * Math.sin(angle),
                    labelX: cx + (radius + 25) * Math.cos(angle),
                    labelY: cy + (radius + 25) * Math.sin(angle),
                    label,
                    value
                }};
            }});
            
            const pathD = points.map((p, i) => `${{i === 0 ? 'M' : 'L'}} ${{p.x}} ${{p.y}}`).join(' ') + ' Z';
            
            // Grid circles
            const gridCircles = [25, 50, 75, 100].map(pct => (
                <circle key={{pct}} cx={{cx}} cy={{cy}} r={{radius * pct / 100}} 
                    fill="none" stroke="#e5e7eb" strokeWidth="1" />
            ));
            
            // Grid lines
            const gridLines = dims.map((_, i) => {{
                const angle = i * angleStep - Math.PI / 2;
                return (
                    <line key={{i}} x1={{cx}} y1={{cy}} 
                        x2={{cx + radius * Math.cos(angle)}} 
                        y2={{cy + radius * Math.sin(angle)}}
                        stroke="#e5e7eb" strokeWidth="1" />
                );
            }});

            return (
                <svg viewBox="0 0 300 300" className="w-full max-w-md mx-auto">
                    {{gridCircles}}
                    {{gridLines}}
                    <path d={{pathD}} fill="rgba(59, 130, 246, 0.25)" stroke="#3b82f6" strokeWidth="2" />
                    {{points.map((p, i) => (
                        <g key={{i}}>
                            <circle cx={{p.x}} cy={{p.y}} r="5" fill="#3b82f6" />
                            <text x={{p.labelX}} y={{p.labelY}} textAnchor="middle" 
                                dominantBaseline="middle" className="text-xs fill-gray-600">
                                {{p.label}}
                            </text>
                            <text x={{p.x}} y={{p.y - 12}} textAnchor="middle" 
                                className="text-xs font-bold fill-blue-600">
                                {{p.value.toFixed(0)}}
                            </text>
                        </g>
                    ))}}
                </svg>
            );
        }};

        // Ideal XI Pitch
        const IdealXIPitch = () => {{
            const positions = {{
                "GK": {{ x: 50, y: 90 }},
                "LB": {{ x: 15, y: 70 }},
                "CB1": {{ x: 35, y: 75 }},
                "CB2": {{ x: 65, y: 75 }},
                "RB": {{ x: 85, y: 70 }},
                "DM": {{ x: 50, y: 55 }},
                "CM": {{ x: 30, y: 45 }},
                "AM": {{ x: 70, y: 45 }},
                "LW": {{ x: 15, y: 25 }},
                "CF": {{ x: 50, y: 15 }},
                "RW": {{ x: 85, y: 25 }}
            }};

            return (
                <div className="relative bg-green-700 rounded-lg p-4" style={{{{aspectRatio: '3/4'}}}}>
                    {{/* Pitch markings */}}
                    <div className="absolute inset-4 border-2 border-white/50 rounded">
                        <div className="absolute top-1/2 left-0 right-0 border-t-2 border-white/50"></div>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border-2 border-white/50 rounded-full"></div>
                    </div>
                    
                    {{/* Players */}}
                    {{idealXI.map((player, i) => {{
                        const pos = positions[player.position];
                        if (!pos) return null;
                        
                        const score = player.fit_score || 0;
                        const bgColor = score >= 75 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-500' : 'bg-orange-500';
                        const name = player.name?.split(' ').pop() || 'Unknown';
                        
                        return (
                            <div key={{i}} className="absolute transform -translate-x-1/2 -translate-y-1/2"
                                style={{{{ left: `${{pos.x}}%`, top: `${{pos.y}}%` }}}}>
                                <div className={{`w-10 h-10 rounded-full ${{bgColor}} border-2 border-white flex items-center justify-center text-white font-bold text-sm shadow-lg`}}>
                                    {{score.toFixed(0)}}
                                </div>
                                <div className="text-white text-xs font-medium mt-1 text-center bg-black/40 px-1 rounded">
                                    {{name}}
                                </div>
                            </div>
                        );
                    }})}}
                </div>
            );
        }};

        // Squad Fit Bar Chart
        const SquadFitChart = () => {{
            const sorted = [...squadFit].sort((a, b) => b.score - a.score).slice(0, 15);
            const maxScore = Math.max(...sorted.map(p => p.score), 100);
            
            return (
                <div className="space-y-2">
                    {{sorted.map((player, i) => (
                        <div key={{i}} className="flex items-center gap-2">
                            <div className="w-32 text-sm truncate">{{player.name}}</div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                                <div className="h-full rounded-full transition-all"
                                    style={{{{
                                        width: `${{(player.score / maxScore) * 100}}%`,
                                        backgroundColor: classificationColors[player.classification]
                                    }}}}>
                                </div>
                                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-medium">
                                    {{player.score.toFixed(0)}}
                                </span>
                            </div>
                        </div>
                    ))}}
                </div>
            );
        }};

        // Recruitment Table
        const RecruitmentTable = () => (
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b">
                        <th className="text-left py-2">Position</th>
                        <th className="text-left py-2">Gap</th>
                        <th className="text-left py-2">Urgency</th>
                        <th className="text-left py-2">Timeline</th>
                        <th className="text-right py-2">Est. Cost</th>
                    </tr>
                </thead>
                <tbody>
                    {{recruitment.map((r, i) => (
                        <tr key={{i}} className="border-b hover:bg-gray-50">
                            <td className="py-2 font-medium">{{r.position}}</td>
                            <td className="py-2">{{r.gap?.toFixed(1)}}</td>
                            <td className="py-2">
                                <span className="px-2 py-1 rounded text-xs text-white"
                                    style={{{{ backgroundColor: urgencyColors[r.urgency] }}}}>
                                    {{r.urgency}}
                                </span>
                            </td>
                            <td className="py-2">{{r.timeline}}</td>
                            <td className="py-2 text-right">£{{r.cost_low}}M - £{{r.cost_high}}M</td>
                        </tr>
                    ))}}
                </tbody>
            </table>
        );

        // Main Dashboard
        const Dashboard = () => {{
            const [tab, setTab] = useState('overview');
            
            const avgFit = squadFit.length > 0 
                ? (squadFit.reduce((a, b) => a + b.score, 0) / squadFit.length).toFixed(1)
                : 0;
            
            const xiAvgFit = idealXI.length > 0
                ? (idealXI.reduce((a, b) => a + (b.fit_score || 0), 0) / idealXI.length).toFixed(1)
                : 0;

            const totalInvestLow = recruitment.reduce((a, b) => a + (b.cost_low || 0), 0);
            const totalInvestHigh = recruitment.reduce((a, b) => a + (b.cost_high || 0), 0);

            return (
                <div className="min-h-screen bg-gray-100">
                    {{/* Header */}}
                    <header className="bg-gradient-to-r from-blue-900 to-blue-700 text-white p-6 shadow-xl">
                        <div className="max-w-7xl mx-auto flex justify-between items-center">
                            <div>
                                <h1 className="text-3xl font-bold">AEGIS Platform</h1>
                                <p className="text-blue-200">Manager Tactical Fit Intelligence (MTFI)</p>
                            </div>
                            <div className="text-right">
                                <p className="text-sm text-blue-200">Scenario Analysis</p>
                                <p className="text-xl font-semibold">{{managerName}}</p>
                            </div>
                        </div>
                    </header>

                    {{/* Navigation */}}
                    <nav className="bg-white shadow-sm sticky top-0 z-10">
                        <div className="max-w-7xl mx-auto flex">
                            {{[
                                {{ id: 'overview', label: '📊 Overview' }},
                                {{ id: 'dna', label: '🧬 Manager DNA' }},
                                {{ id: 'squad', label: '👥 Squad Fit' }},
                                {{ id: 'recruitment', label: '🎯 Recruitment' }}
                            ].map(t => (
                                <button key={{t.id}} onClick={{() => setTab(t.id)}}
                                    className={{`px-6 py-4 text-sm font-medium transition-colors
                                        ${{tab === t.id ? 'tab-active bg-blue-50' : 'text-gray-600 hover:bg-gray-50'}}`}}>
                                    {{t.label}}
                                </button>
                            ))}}
                        </div>
                    </nav>

                    {{/* Content */}}
                    <main className="max-w-7xl mx-auto p-6">
                        {{tab === 'overview' && (
                            <div className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <p className="text-gray-500 text-sm">Matches Analysed</p>
                                        <p className="text-3xl font-bold text-blue-600">{{matchesAnalysed}}</p>
                                    </div>
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <p className="text-gray-500 text-sm">Primary Formation</p>
                                        <p className="text-3xl font-bold text-blue-600">{{primaryFormation}}</p>
                                    </div>
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <p className="text-gray-500 text-sm">Squad Avg Fit</p>
                                        <p className="text-3xl font-bold text-blue-600">{{avgFit}}</p>
                                    </div>
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <p className="text-gray-500 text-sm">Key Enablers</p>
                                        <p className="text-3xl font-bold text-green-600">{{squadSummary.key_enablers || 0}}</p>
                                    </div>
                                </div>
                                
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <h2 className="text-lg font-bold mb-4">Manager DNA Profile</h2>
                                        <DNARadar />
                                    </div>
                                    <div className="bg-white rounded-xl p-6 shadow-sm">
                                        <h2 className="text-lg font-bold mb-4">Ideal XI ({{primaryFormation}})</h2>
                                        <IdealXIPitch />
                                        <p className="text-center text-sm text-gray-500 mt-2">Average Fit: {{xiAvgFit}}</p>
                                    </div>
                                </div>

                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Squad Classification</h2>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        {{Object.entries(classificationColors).map(([cls, color]) => {{
                                            const count = squadFit.filter(p => p.classification === cls).length;
                                            return (
                                                <div key={{cls}} className="text-center p-4 rounded-lg" 
                                                    style={{{{ backgroundColor: color + '20' }}}}>
                                                    <p className="text-3xl font-bold" style={{{{ color }}}}>{{count}}</p>
                                                    <p className="text-sm text-gray-600">{{cls}}</p>
                                                </div>
                                            );
                                        }})}}
                                    </div>
                                </div>
                            </div>
                        )}}

                        {{tab === 'dna' && (
                            <div className="space-y-6">
                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Manager DNA Radar</h2>
                                    <DNARadar />
                                </div>
                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">DNA Dimensions</h2>
                                    <div className="space-y-3">
                                        {{Object.entries(dnaDimensions).map(([dim, value]) => (
                                            <div key={{dim}} className="flex items-center gap-4">
                                                <div className="w-40 font-medium">{{dim}}</div>
                                                <div className="flex-1 bg-gray-200 rounded-full h-4">
                                                    <div className="bg-blue-500 h-full rounded-full"
                                                        style={{{{ width: `${{value}}%` }}}}>
                                                    </div>
                                                </div>
                                                <div className="w-12 text-right font-bold">{{value.toFixed(0)}}</div>
                                            </div>
                                        ))}}
                                    </div>
                                </div>
                            </div>
                        )}}

                        {{tab === 'squad' && (
                            <div className="space-y-6">
                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Ideal Starting XI</h2>
                                    <IdealXIPitch />
                                </div>
                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Squad Fit Scores (Top 15)</h2>
                                    <SquadFitChart />
                                </div>
                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Full Squad</h2>
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b">
                                                <th className="text-left py-2">Player</th>
                                                <th className="text-left py-2">Position</th>
                                                <th className="text-left py-2">Age</th>
                                                <th className="text-left py-2">Fit Score</th>
                                                <th className="text-left py-2">Classification</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {{[...squadFit].sort((a, b) => b.score - a.score).map((p, i) => (
                                                <tr key={{i}} className="border-b hover:bg-gray-50">
                                                    <td className="py-2 font-medium">{{p.name}}</td>
                                                    <td className="py-2">{{p.position}}</td>
                                                    <td className="py-2">{{p.age}}</td>
                                                    <td className="py-2 font-bold">{{p.score.toFixed(0)}}</td>
                                                    <td className="py-2">
                                                        <span className="px-2 py-1 rounded text-xs text-white"
                                                            style={{{{ backgroundColor: classificationColors[p.classification] }}}}>
                                                            {{p.classification}}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}}

                        {{tab === 'recruitment' && (
                            <div className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                                        <h3 className="font-bold text-red-800 mb-2">🔴 January Window</h3>
                                        <p className="text-3xl font-bold text-red-600">
                                            {{recruitment.filter(r => r.timeline === 'January').length}} Signings
                                        </p>
                                        <ul className="mt-2 space-y-1 text-sm">
                                            {{recruitment.filter(r => r.timeline === 'January').map((r, i) => (
                                                <li key={{i}} className="flex justify-between">
                                                    <span>{{r.position}}</span>
                                                    <span className="font-medium">£{{r.cost_low}}M - £{{r.cost_high}}M</span>
                                                </li>
                                            ))}}
                                        </ul>
                                    </div>
                                    <div className="bg-orange-50 border border-orange-200 rounded-xl p-6">
                                        <h3 className="font-bold text-orange-800 mb-2">🟠 Summer Window</h3>
                                        <p className="text-3xl font-bold text-orange-600">
                                            {{recruitment.filter(r => r.timeline === 'Summer').length}} Signings
                                        </p>
                                        <ul className="mt-2 space-y-1 text-sm">
                                            {{recruitment.filter(r => r.timeline === 'Summer').map((r, i) => (
                                                <li key={{i}} className="flex justify-between">
                                                    <span>{{r.position}}</span>
                                                    <span className="font-medium">£{{r.cost_low}}M - £{{r.cost_high}}M</span>
                                                </li>
                                            ))}}
                                        </ul>
                                    </div>
                                </div>

                                <div className="bg-white rounded-xl p-6 shadow-sm">
                                    <h2 className="text-lg font-bold mb-4">Recruitment Priorities</h2>
                                    <RecruitmentTable />
                                </div>

                                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                                    <h3 className="font-bold text-blue-800 mb-2">💰 Total Investment Required</h3>
                                    <p className="text-4xl font-bold text-blue-600">
                                        £{{totalInvestLow}}M - £{{totalInvestHigh}}M
                                    </p>
                                    <p className="text-sm text-blue-600 mt-1">
                                        Across {{recruitment.length}} priority positions
                                    </p>
                                </div>
                            </div>
                        )}}
                    </main>

                    {{/* Footer */}}
                    <footer className="bg-gray-800 text-gray-400 p-6 mt-12">
                        <div className="max-w-7xl mx-auto text-center">
                            <p className="text-sm">
                                <span className="text-white font-medium">Aegis Football Advisory Group</span> | 
                                Manager Tactical Fit Intelligence
                            </p>
                            <p className="text-xs mt-2">Data source: Sportsmonks API</p>
                        </div>
                    </footer>
                </div>
            );
        }};

        ReactDOM.render(<Dashboard />, document.getElementById('root'));
    </script>
</body>
</html>'''
        
        return html

    # =========================================================================
    # DNA RADAR
    # =========================================================================
    
    def plot_dna_radar(self):
        """Generate Manager DNA radar chart."""
        import numpy as np
        
        dimensions = self.results.get("dna_dimensions", {})
        labels = list(dimensions.keys())
        values = list(dimensions.values())
        
        # Complete the loop
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = self.plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot
        ax.fill(angles, values, color=self.COLORS["primary"], alpha=0.25)
        ax.plot(angles, values, color=self.COLORS["primary"], linewidth=2)
        ax.scatter(angles[:-1], values[:-1], color=self.COLORS["primary"], s=80, zorder=5)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11)
        ax.set_ylim(0, 100)
        
        # Title
        manager = self.results.get("manager", "Unknown")
        ax.set_title(f"Manager DNA: {manager}", size=16, fontweight="bold", pad=20)
        
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "01_manager_dna_radar.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 01_manager_dna_radar.png")
    
    # =========================================================================
    # FORMATION USAGE
    # =========================================================================
    
    def plot_formation_usage(self):
        """Generate formation usage bar chart."""
        # This would need formation data from manager_dna
        # For now, create placeholder
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        formations = ["4-3-3", "3-5-2", "4-2-3-1", "3-4-3", "5-3-2"]
        usage = [39.5, 26.3, 17.1, 10.5, 6.6]  # Example
        
        bars = ax.barh(formations, usage, color=self.COLORS["primary"])
        
        # Add percentage labels
        for bar, pct in zip(bars, usage):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f"{pct}%", va="center", fontweight="bold")
        
        ax.set_xlabel("Usage %")
        ax.set_title("Formation Usage", fontweight="bold", size=14)
        ax.set_xlim(0, 50)
        
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "02_formation_usage.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 02_formation_usage.png")
    
    # =========================================================================
    # SQUAD FIT
    # =========================================================================
    
    def plot_squad_fit(self):
        """Generate squad fit score distribution."""
        # Load squad fit CSV
        squad_file = self.output_dir / "squad_fit_scores.csv"
        players = []
        if squad_file.exists():
            with open(squad_file) as f:
                reader = csv.DictReader(f)
                players = list(reader)
        
        if not players:
            print("  ⚠ No squad fit data")
            return
        
        # Sort by score
        players.sort(key=lambda x: float(x.get("Fit Score", 0)), reverse=True)
        
        fig, ax = self.plt.subplots(figsize=(12, max(8, len(players) * 0.4)))
        
        names = [p["Name"] for p in players]
        scores = [float(p["Fit Score"]) for p in players]
        classifications = [p["Classification"] for p in players]
        colors = [self.COLORS.get(c, self.COLORS["secondary"]) for c in classifications]
        
        bars = ax.barh(names, scores, color=colors)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f"{score:.0f}", va="center", fontsize=9)
        
        ax.set_xlabel("Fit Score")
        ax.set_xlim(0, 110)
        ax.set_title("Squad Fit Scores", fontweight="bold", size=14)
        ax.invert_yaxis()
        
        # Legend
        handles = [self.mpatches.Patch(color=self.COLORS[c], label=c) 
                  for c in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]]
        ax.legend(handles=handles, loc="lower right")
        
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "03_squad_fit_scores.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 03_squad_fit_scores.png")
    
    # =========================================================================
    # CLASSIFICATION PIE
    # =========================================================================
    
    def plot_classification_pie(self):
        """Generate classification distribution pie chart."""
        summary = self.results.get("squad_summary", {})
        
        labels = ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]
        sizes = [
            summary.get("key_enablers", 0),
            summary.get("good_fit", 0),
            summary.get("system_dependent", 0),
            summary.get("marginalised", 0)
        ]
        colors = [self.COLORS[l] for l in labels]
        
        # Filter out zeros
        data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if not data:
            return
        
        labels, sizes, colors = zip(*data)
        
        fig, ax = self.plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 11}
        )
        
        ax.set_title("Squad Classification", fontweight="bold", size=14)
        
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "04_classification_pie.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 04_classification_pie.png")
    
    # =========================================================================
    # IDEAL XI PITCH
    # =========================================================================
    
    def plot_ideal_xi(self):
        """Generate ideal XI pitch visualization."""
        ideal_xi = self.results.get("ideal_xi", [])
        
        # 4-3-3 positions (x, y coordinates on pitch)
        positions = {
            "GK": (50, 92),
            "LB": (15, 70),
            "CB1": (35, 75),
            "CB2": (65, 75),
            "RB": (85, 70),
            "DM": (50, 55),
            "CM": (30, 45),
            "AM": (70, 45),
            "LW": (15, 25),
            "CF": (50, 15),
            "RW": (85, 25)
        }
        
        fig, ax = self.plt.subplots(figsize=(10, 14))
        
        # Draw pitch
        ax.set_facecolor("#2d5a27")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Pitch markings
        ax.axhline(y=50, color="white", linewidth=2, alpha=0.7)
        
        # Centre circle
        circle = self.plt.Circle((50, 50), 10, fill=False, color="white", linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        
        # Penalty areas
        ax.add_patch(self.plt.Rectangle((25, 82), 50, 18, fill=False, color="white", linewidth=2, alpha=0.7))
        ax.add_patch(self.plt.Rectangle((25, 0), 50, 18, fill=False, color="white", linewidth=2, alpha=0.7))
        
        # Plot players
        for player in ideal_xi:
            pos_key = player.get("position", "")
            if pos_key in positions:
                x, y = positions[pos_key]
                score = player.get("fit_score", 0)
                
                # Colour based on score
                if score >= 75:
                    color = self.COLORS["Key Enabler"]
                elif score >= 60:
                    color = self.COLORS["Good Fit"]
                else:
                    color = self.COLORS["System Dependent"]
                
                # Player circle
                circle = self.plt.Circle((x, y), 5, color=color, ec="white", linewidth=2, zorder=5)
                ax.add_patch(circle)
                
                # Score in circle
                ax.text(x, y, f"{score:.0f}", ha="center", va="center",
                       color="white", fontweight="bold", fontsize=10, zorder=6)
                
                # Name below
                name = player.get("name", "").split()[-1]  # Last name only
                ax.text(x, y - 8, name, ha="center", va="center",
                       color="white", fontsize=9, fontweight="bold",
                       bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
        
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Ideal XI (4-3-3)", fontweight="bold", size=16, color="white", pad=20)
        
        fig.patch.set_facecolor("#2d5a27")
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "05_ideal_xi_pitch.png", dpi=150, 
                        bbox_inches="tight", facecolor="#2d5a27")
        self.plt.close()
        print("  ✓ 05_ideal_xi_pitch.png")
    
    # =========================================================================
    # RECRUITMENT
    # =========================================================================
    
    def plot_recruitment(self):
        """Generate recruitment priorities chart."""
        recruitment = self.results.get("recruitment", [])
        
        if not recruitment:
            print("  ⚠ No recruitment priorities")
            return
        
        fig, ax = self.plt.subplots(figsize=(12, max(6, len(recruitment) * 0.8)))
        
        positions = [r["position"] for r in recruitment]
        gaps = [r["gap"] for r in recruitment]
        urgencies = [r["urgency"] for r in recruitment]
        
        colors = {
            "Critical": "#ef4444",
            "High": "#f97316",
            "Medium": "#eab308"
        }
        bar_colors = [colors.get(u, "#64748b") for u in urgencies]
        
        bars = ax.barh(positions, gaps, color=bar_colors)
        
        # Add cost labels
        for bar, r in zip(bars, recruitment):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f"£{r['cost_low']}M-£{r['cost_high']}M",
                   va="center", fontsize=10)
        
        ax.set_xlabel("Fit Gap (points below 75)")
        ax.set_title("Recruitment Priorities", fontweight="bold", size=14)
        ax.set_xlim(0, max(gaps) * 1.5 if gaps else 30)
        ax.invert_yaxis()
        
        # Legend
        handles = [self.mpatches.Patch(color=colors[u], label=u) for u in ["Critical", "High", "Medium"]]
        ax.legend(handles=handles, loc="lower right")
        
        self.plt.tight_layout()
        self.plt.savefig(self.output_dir / "06_recruitment_priorities.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 06_recruitment_priorities.png")
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    
    def plot_executive_summary(self):
        """Generate executive summary dashboard."""
        import numpy as np
        
        fig = self.plt.figure(figsize=(16, 12))
        
        manager = self.results.get("manager", "Unknown")
        fig.suptitle(f"Aegis Analysis: {manager}", fontsize=20, fontweight="bold", y=0.98)
        
        # Grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. DNA Dimensions (bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        dims = self.results.get("dna_dimensions", {})
        ax1.barh(list(dims.keys()), list(dims.values()), color=self.COLORS["primary"])
        ax1.set_xlim(0, 100)
        ax1.set_title("Manager DNA", fontweight="bold")
        
        # 2. Classification (pie)
        ax2 = fig.add_subplot(gs[0, 1])
        summary = self.results.get("squad_summary", {})
        sizes = [summary.get("key_enablers", 0), summary.get("good_fit", 0),
                summary.get("system_dependent", 0), summary.get("marginalised", 0)]
        labels = ["Key Enabler", "Good Fit", "System Dep.", "Marginalised"]
        colors = [self.COLORS[l.replace(" Dep.", " Dependent").replace("Marginalised", "Potentially Marginalised")] 
                 for l in labels]
        
        # Filter zeros
        data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if data:
            labels, sizes, colors = zip(*data)
            ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%")
        ax2.set_title("Squad Classification", fontweight="bold")
        
        # 3. Key Metrics
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")
        metrics_text = f"""
        Matches Analysed: {self.results.get('matches_analysed', 0)}
        
        Squad Size: {summary.get('total', 0)}
        
        Average Fit: {summary.get('average_fit', 0):.1f}
        
        Key Enablers: {summary.get('key_enablers', 0)}
        """
        ax3.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment="center",
                fontfamily="monospace")
        ax3.set_title("Key Metrics", fontweight="bold")
        
        # 4. Top 5 Fits
        ax4 = fig.add_subplot(gs[1, 0])
        ideal = self.results.get("ideal_xi", [])[:5]
        if ideal:
            names = [p["name"].split()[-1] for p in ideal]
            scores = [p["fit_score"] for p in ideal]
            ax4.barh(names, scores, color=self.COLORS["Key Enabler"])
            ax4.set_xlim(0, 100)
        ax4.set_title("Top 5 Fit Scores", fontweight="bold")
        ax4.invert_yaxis()
        
        # 5. Recruitment
        ax5 = fig.add_subplot(gs[1, 1])
        recruitment = self.results.get("recruitment", [])[:4]
        if recruitment:
            positions = [r["position"] for r in recruitment]
            gaps = [r["gap"] for r in recruitment]
            ax5.barh(positions, gaps, color="#f97316")
        ax5.set_title("Top Recruitment Needs", fontweight="bold")
        ax5.invert_yaxis()
        
        # 6. Investment Summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        recruitment_full = self.results.get("recruitment", [])
        total_low = sum(r.get("cost_low", 0) for r in recruitment_full)
        total_high = sum(r.get("cost_high", 0) for r in recruitment_full)
        
        investment_text = f"""
        Total Investment Required:
        
        £{total_low}M - £{total_high}M
        
        Priority Signings: {len(recruitment_full)}
        """
        ax6.text(0.1, 0.5, investment_text, fontsize=14, verticalalignment="center",
                fontfamily="monospace")
        ax6.set_title("Investment Summary", fontweight="bold")
        
        self.plt.savefig(self.output_dir / "07_executive_summary.png", dpi=150, bbox_inches="tight")
        self.plt.close()
        print("  ✓ 07_executive_summary.png")


# =============================================================================
# STANDALONE CONVENIENCE FUNCTION
# =============================================================================

def generate_mtfi_dashboard(
    data_dir: str = ".",
    output_path: str = "MTFI_Dashboard.html",
    manager_name: Optional[str] = None,
    target_club: Optional[str] = None,
    season: str = "2024/25"
) -> str:
    """
    Generate MTFI Dashboard v2 HTML from CSV data files.
    
    This is a convenience function that wraps AegisVisualizer.generate_dashboard_v2().
    
    Args:
        data_dir: Directory containing CSV files (default: current directory)
        output_path: Path for output HTML file
        manager_name: Override manager name (auto-detected if None)
        target_club: Override target club name (auto-detected if None)
        season: Season label for display
    
    Returns:
        Path to generated HTML file
    
    Required CSV files in data_dir:
        - manager_profiles.csv (from ManagerDNATrainer)
        - cluster_centroids.csv (from ManagerDNATrainer)
        - squad_fit_scores.csv (from SquadFitAnalyzer or AegisAnalyzer)
        - ideal_xi.csv (from SquadFitAnalyzer)
        - recruitment_priorities.csv (from AegisAnalyzer)
    
    Example:
        from aegis.visualizations import generate_mtfi_dashboard
        
        # Generate from outputs directory
        generate_mtfi_dashboard(
            data_dir="/content/aegis_data/outputs",
            output_path="/content/MTFI_Dashboard.html"
        )
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    
    # Create visualizer pointing to data directory
    viz = AegisVisualizer(output_dir=data_dir)
    
    # Generate dashboard - output to specified path
    original_output_dir = viz.output_dir
    viz.output_dir = output_path.parent
    
    result = viz.generate_dashboard_v2(
        filename=output_path.name,
        manager_name=manager_name,
        target_club=target_club,
        season=season
    )
    
    # Restore
    viz.output_dir = original_output_dir
    
    return str(result)