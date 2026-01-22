"""
Aegis Visualizations
====================
Generate charts and dashboards for analysis results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

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
    
    def generate_dashboard(self, filename: str = "AEGIS_Dashboard.html"):
        """
        Generate standalone interactive HTML dashboard.
        
        Creates a single HTML file with embedded React that opens
        in any browser - no development environment needed.
        
        Args:
            filename: Output filename (default: AEGIS_Dashboard.html)
        
        Returns:
            Path to generated dashboard file
        """
        if not self.results:
            raise ValueError("No results loaded. Call load_results() first.")
        
        print("\nGenerating interactive dashboard...")
        
        # Prepare data for embedding
        manager = self.results.get("manager", "Unknown")
        primary_formation = self.results.get("primary_formation", "4-3-3")
        matches = self.results.get("matches_analysed", 0)
        dna_dimensions = self.results.get("dna_dimensions", {})
        squad_summary = self.results.get("squad_summary", {})
        ideal_xi = self.results.get("ideal_xi", [])
        recruitment = self.results.get("recruitment", [])
        
        # Load squad fit data
        squad_fit = []
        if self.squad_fit_data:
            for p in self.squad_fit_data:
                squad_fit.append({
                    "name": p.get("Name", "Unknown"),
                    "position": p.get("Position", "Unknown"),
                    "age": int(p.get("Age", 25)),
                    "score": float(p.get("Fit Score", 0)),
                    "classification": p.get("Classification", "Unknown")
                })
        
        # Generate HTML
        html_content = self._generate_dashboard_html(
            manager=manager,
            primary_formation=primary_formation,
            matches=matches,
            dna_dimensions=dna_dimensions,
            squad_summary=squad_summary,
            squad_fit=squad_fit,
            ideal_xi=ideal_xi,
            recruitment=recruitment
        )
        
        # Save
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✓ Dashboard saved: {output_path}")
        return output_path
    
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