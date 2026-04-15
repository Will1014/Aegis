"""
Aegis Aegis Complete Analysis Engine
====================================
Manager DNA, Squad Fit, and Recruitment Analysis

Works with the ETL-processed data structure.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from math import pi
from typing import Dict, List

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AegisAnalyzer:
    """Complete Aegis Analysis: Manager DNA → Squad Fit → Recruitment"""
    
    def __init__(self):
        self.manager_dna = None
        self.squad_df = None
        self.fit_results = None
        self.ideal_xi = None
        self.recruitment_priorities = None
        
    def load_data(self):
        """Load ETL-processed data"""
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        with open(PROCESSED_DIR / "manager_dna.json") as f:
            self.manager_dna = json.load(f)
        
        self.squad_df = pd.read_csv(PROCESSED_DIR / "squad_profiles.csv")
        
        print(f"✓ Manager: {self.manager_dna['manager_name']}")
        print(f"✓ Matches Analyzed: {self.manager_dna['matches_analyzed']}")
        print(f"✓ Squad Size: {len(self.squad_df)} players")
        return self
    
    # =========================================================================
    # MANAGER DNA DIMENSIONS
    # =========================================================================
    
    def calculate_dna_dimensions(self) -> 'AegisAnalyzer':
        """Calculate 8 tactical dimensions (0-100 scale)"""
        
        tactical = self.manager_dna["tactical_profile"]
        results = self.manager_dna["results_profile"]
        formation = self.manager_dna["formation_profile"]
        
        poss = tactical["possession"]["avg"]
        press = tactical["pressing"]["intensity"]["avg"]
        pass_acc = tactical["build_up"]["pass_accuracy"]
        shots = tactical["attacking"]["shots_pg"]
        
        dimensions = {
            "Possession": min(100, max(0, (poss - 35) * 2.5)),
            "Pressing": min(100, max(0, (press - 15) * 4)),
            "Build-up": min(100, pass_acc),
            "Attacking Threat": min(100, shots * 6),
            "Defensive Solidity": min(100, 100 - results["conceded_per_game"] * 30),
            "Set Piece": 50,  # Would need event data
            "Flexibility": min(100, formation["flexibility_score"] * 20),
            "Results": min(100, results["points_per_game"] * 33)
        }
        
        self.dna_dimensions = {k: round(v, 1) for k, v in dimensions.items()}
        return self
    
    # =========================================================================
    # SQUAD FIT SCORING
    # =========================================================================
    
    def calculate_squad_fit(self):
        """Calculate fit scores for all players"""
        
        print("\n" + "=" * 60)
        print("CALCULATING SQUAD FIT")
        print("=" * 60)
        
        results = []
        
        for _, player in self.squad_df.iterrows():
            fit_score = self._calculate_player_fit(player)
            classification = self._classify_player(fit_score)
            
            results.append({
                "player_id": player["player_id"],
                "name": player["name"],
                "age": player["age"],
                "position": player["position"],
                "detailed_position": player["detailed_position"],
                "market_value": player.get("market_value", 0),
                "fit_score": fit_score,
                "classification": classification
            })
        
        self.fit_results = pd.DataFrame(results).sort_values("fit_score", ascending=False)
        
        # Print summary
        print("\nFIT CLASSIFICATIONS:")
        for cls in ["Key Enabler", "Good Fit", "System Dependent", "Potentially Marginalised"]:
            count = len(self.fit_results[self.fit_results["classification"] == cls])
            players = self.fit_results[self.fit_results["classification"] == cls]["name"].tolist()
            print(f"  {cls}: {count} players")
            for p in players[:3]:
                print(f"    • {p}")
            if len(players) > 3:
                print(f"    ... and {len(players)-3} more")
        
        return self
    
    def _calculate_player_fit(self, player) -> float:
        """Calculate fit score for a single player"""
        
        position = str(player.get("detailed_position", player.get("position", "")))
        
        # Base score from rating-equivalent metrics
        base_score = 60
        
        # Position-specific adjustments
        if "Goalkeeper" in position:
            saves = player.get("saves", 0)
            clean_sheets = player.get("clean_sheets", 0)
            base_score = min(100, 50 + saves * 0.3 + clean_sheets * 3)
            
        elif "Back" in position or "Centre-Back" in position:
            tackles = player.get("tackles", 0)
            interceptions = player.get("interceptions", 0)
            clearances = player.get("clearances", 0)
            pass_acc = player.get("pass_accuracy", 75)
            
            # High press system values active defenders
            defensive_score = min(100, (tackles + interceptions) * 0.8)
            passing_score = min(100, (pass_acc - 70) * 3)
            base_score = defensive_score * 0.5 + passing_score * 0.3 + 40 * 0.2
            
        elif "Midfield" in position:
            tackles = player.get("tackles", 0)
            interceptions = player.get("interceptions", 0)
            key_passes = player.get("key_passes", 0)
            dribbles = player.get("successful_dribbles", 0)
            goals = player.get("goals", 0)
            assists = player.get("assists", 0)
            
            if "Attacking" in position:
                base_score = min(100, (goals + assists) * 4 + key_passes * 0.8 + dribbles * 0.5)
            elif "Defensive" in position:
                base_score = min(100, (tackles + interceptions) * 0.7 + 40)
            else:
                base_score = min(100, (tackles + interceptions) * 0.4 + (goals + assists) * 2 + key_passes * 0.5 + 30)
            
        elif "Winger" in position or "Forward" in position:
            goals = player.get("goals", 0)
            assists = player.get("assists", 0)
            shots_ot = player.get("shots_on_target", 0)
            dribbles = player.get("successful_dribbles", 0)
            key_passes = player.get("key_passes", 0)
            
            base_score = min(100, goals * 4 + assists * 3 + shots_ot * 0.5 + dribbles * 0.3 + key_passes * 0.4)
        
        # Age modifier (high-press favors younger players)
        age = player.get("age", 25)
        if age > 30:
            base_score *= (1 - (age - 30) * 0.015)
        elif age < 22:
            base_score *= 0.95
        
        return round(max(0, min(100, base_score)), 1)
    
    def _classify_player(self, score: float) -> str:
        if score >= 75: return "Key Enabler"
        if score >= 60: return "Good Fit"
        if score >= 45: return "System Dependent"
        return "Potentially Marginalised"
    
    # =========================================================================
    # IDEAL XI
    # =========================================================================
    
    def generate_ideal_xi(self):
        """Generate Ideal XI for 4-3-3"""
        
        print("\n" + "=" * 60)
        print("GENERATING IDEAL XI")
        print("=" * 60)
        
        positions_needed = [
            ("Goalkeeper", 1),
            ("Centre-Back", 2),
            ("Right-Back", 1),
            ("Left-Back", 1),
            ("Defensive Midfield", 1),
            ("Central Midfield", 1),
            ("Attacking Midfield", 1),
            ("Left Winger", 1),
            ("Right Winger", 1),
            ("Centre-Forward", 1)
        ]
        
        ideal_xi = []
        used = set()
        
        for position, count in positions_needed:
            available = self.fit_results[
                (self.fit_results["detailed_position"].str.contains(position, na=False)) &
                (~self.fit_results["player_id"].isin(used))
            ].head(count)
            
            for _, player in available.iterrows():
                ideal_xi.append({
                    "position": position,
                    "name": player["name"],
                    "fit_score": player["fit_score"],
                    "classification": player["classification"]
                })
                used.add(player["player_id"])
        
        self.ideal_xi = pd.DataFrame(ideal_xi)
        
        print(f"\nIDEAL XI (4-3-3):")
        print("-" * 50)
        for _, p in self.ideal_xi.iterrows():
            emoji = "🟢" if p["classification"] == "Key Enabler" else "🟡" if p["classification"] == "Good Fit" else "🟠"
            print(f"  {p['position']:<22} {p['name']:<20} {emoji} {p['fit_score']:.0f}")
        
        avg_fit = self.ideal_xi["fit_score"].mean()
        print(f"\nAverage XI Fit: {avg_fit:.1f}")
        
        return self
    
    # =========================================================================
    # RECRUITMENT ANALYSIS
    # =========================================================================
    
    def analyze_recruitment(self):
        """Identify recruitment priorities"""
        
        print("\n" + "=" * 60)
        print("RECRUITMENT ANALYSIS")
        print("=" * 60)
        
        # Calculate position gaps
        position_fit = self.fit_results.groupby("detailed_position").agg({
            "fit_score": ["mean", "count"],
            "market_value": "sum"
        })
        position_fit.columns = ["avg_fit", "count", "value"]
        position_fit["gap"] = 75 - position_fit["avg_fit"]
        position_fit = position_fit.sort_values("gap", ascending=False)
        
        # Base costs by position
        base_costs = {
            "Centre-Forward": 50, "Left Winger": 40, "Right Winger": 40,
            "Attacking Midfield": 45, "Central Midfield": 35, "Defensive Midfield": 40,
            "Centre-Back": 35, "Right-Back": 30, "Left-Back": 30, "Goalkeeper": 25
        }
        
        priorities = []
        rank = 0
        
        for position, row in position_fit.iterrows():
            if row["gap"] > 0:
                rank += 1
                urgency = "Critical" if row["gap"] > 20 else "High" if row["gap"] > 10 else "Medium"
                timeline = "January" if urgency == "Critical" else "Summer"
                
                base_cost = base_costs.get(position, 30)
                multiplier = 1.3 if urgency == "Critical" else 1.1
                
                priorities.append({
                    "rank": rank,
                    "position": position,
                    "urgency": urgency,
                    "timeline": timeline,
                    "current_fit": round(row["avg_fit"], 1),
                    "gap": round(row["gap"], 1),
                    "cost_low": int(base_cost * 0.7 * multiplier),
                    "cost_high": int(base_cost * 1.3 * multiplier)
                })
        
        self.recruitment_priorities = pd.DataFrame(priorities[:8])
        
        print("\nRECRUITMENT PRIORITIES:")
        print("-" * 70)
        for _, p in self.recruitment_priorities.iterrows():
            emoji = "🔴" if p["urgency"] == "Critical" else "🟠" if p["urgency"] == "High" else "🟡"
            print(f"{p['rank']:2}. {emoji} {p['position']:<22} {p['urgency']:<10} £{p['cost_low']}M-£{p['cost_high']}M")
        
        total_low = self.recruitment_priorities["cost_low"].sum()
        total_high = self.recruitment_priorities["cost_high"].sum()
        print(f"\nTotal Investment: £{total_low}M - £{total_high}M")
        
        return self
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    
    def create_all_visualizations(self):
        """Generate all charts and dashboards"""
        
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self._create_dna_radar()
        self._create_formation_chart()
        self._create_fit_distribution()
        self._create_classification_pie()
        self._create_pitch_xi()
        self._create_recruitment_chart()
        self._create_summary_dashboard()
        
        return self
    
    def _create_dna_radar(self):
        """Create Manager DNA radar chart"""
        
        dims = self.dna_dimensions
        categories = list(dims.keys())
        values = list(dims.values())
        N = len(categories)
        
        angles = [n / N * 2 * pi for n in range(N)]
        values_plot = values + values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='#1e88e5', markersize=8)
        ax.fill(angles, values_plot, alpha=0.25, color='#1e88e5')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80])
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"Manager DNA: {self.manager_dna['manager_name']}\n", size=16, fontweight='bold', y=1.08)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "01_manager_dna_radar.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Manager DNA radar saved")
    
    def _create_formation_chart(self):
        """Create formation usage chart"""
        
        formations = self.manager_dna["formation_profile"]["formation_percentages"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(list(formations.keys()), list(formations.values()), color='#1e88e5', edgecolor='white')
        
        for bar, pct in zip(bars, formations.values()):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{pct}%', va='center', fontweight='bold')
        
        ax.set_xlabel('Usage %')
        ax.set_title(f"Formation Usage: {self.manager_dna['manager_name']}", fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "02_formation_usage.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Formation chart saved")
    
    def _create_fit_distribution(self):
        """Create squad fit distribution chart"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        df = self.fit_results.sort_values("fit_score", ascending=True)
        colors = {"Key Enabler": "#22c55e", "Good Fit": "#eab308", "System Dependent": "#f97316", "Potentially Marginalised": "#ef4444"}
        bar_colors = [colors[c] for c in df["classification"]]
        
        y_pos = range(len(df))
        bars = ax.barh(y_pos, df["fit_score"], color=bar_colors, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['name']} ({row['detailed_position'][:15]})" for _, row in df.iterrows()], fontsize=9)
        ax.set_xlabel("Fit Score")
        ax.set_title(f"Squad Fit: Tottenham → {self.manager_dna['manager_name']}", fontweight='bold')
        ax.set_xlim(0, 100)
        ax.axvline(x=75, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=60, color='gold', linestyle='--', alpha=0.5)
        ax.axvline(x=45, color='orange', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_squad_fit_scores.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Fit distribution saved")
    
    def _create_classification_pie(self):
        """Create classification pie chart"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        counts = self.fit_results["classification"].value_counts()
        colors = {"Key Enabler": "#22c55e", "Good Fit": "#eab308", "System Dependent": "#f97316", "Potentially Marginalised": "#ef4444"}
        pie_colors = [colors.get(c, "#666") for c in counts.index]
        
        wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, colors=pie_colors,
                                          autopct='%1.0f%%', startangle=90, explode=[0.05 if c == "Key Enabler" else 0 for c in counts.index])
        plt.setp(autotexts, size=12, weight='bold', color='white')
        ax.set_title("Squad Compatibility Classification", fontweight='bold')
        
        centre = plt.Circle((0, 0), 0.4, fc='white')
        ax.add_artist(centre)
        ax.text(0, 0, f"{len(self.fit_results)}\nPlayers", ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_classification_pie.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Classification pie saved")
    
    def _create_pitch_xi(self):
        """Create pitch visualization with Ideal XI"""
        
        fig, ax = plt.subplots(figsize=(12, 16))
        pitch_color = '#1a472a'
        ax.set_facecolor(pitch_color)
        
        # Draw pitch
        ax.plot([0, 100, 100, 0, 0], [0, 0, 130, 130, 0], color='white', linewidth=2)
        ax.plot([0, 100], [65, 65], color='white', linewidth=2)
        circle = plt.Circle((50, 65), 9.15, fill=False, color='white', linewidth=2)
        ax.add_patch(circle)
        ax.plot([18, 18, 82, 82], [0, 16.5, 16.5, 0], color='white', linewidth=2)
        ax.plot([18, 18, 82, 82], [130, 113.5, 113.5, 130], color='white', linewidth=2)
        
        # 4-3-3 positions
        positions_433 = {
            "Goalkeeper": (50, 5), "Left-Back": (15, 25), "Centre-Back_1": (35, 22),
            "Centre-Back_2": (65, 22), "Right-Back": (85, 25), "Defensive Midfield": (50, 45),
            "Central Midfield": (30, 55), "Attacking Midfield": (70, 55),
            "Left Winger": (15, 85), "Centre-Forward": (50, 95), "Right Winger": (85, 85)
        }
        
        # Map players to positions
        position_to_player = {}
        cb_count = 0
        for _, player in self.ideal_xi.iterrows():
            pos = player["position"]
            if pos == "Centre-Back":
                cb_count += 1
                pos = f"Centre-Back_{cb_count}"
            position_to_player[pos] = player
        
        colors = {"Key Enabler": "#22c55e", "Good Fit": "#eab308", "System Dependent": "#f97316", "Potentially Marginalised": "#ef4444"}
        
        for pos_key, (x, y) in positions_433.items():
            if pos_key in position_to_player:
                player = position_to_player[pos_key]
                color = colors.get(player["classification"], "#666")
                short_name = player["name"].split()[-1]
                score = player["fit_score"]
            else:
                color = "#666"
                short_name = "TBD"
                score = 0
            
            circle = plt.Circle((x, y), 5, color=color, ec='white', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y - 8, short_name, ha='center', va='top', fontsize=9, color='white', fontweight='bold')
            ax.text(x, y, f"{score:.0f}", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        ax.set_title(f"Ideal XI: {self.manager_dna['manager_name']}'s 4-3-3", fontsize=16, fontweight='bold', color='white', y=1.02)
        
        legend_elements = [
            mpatches.Patch(facecolor='#22c55e', edgecolor='white', label='Key Enabler'),
            mpatches.Patch(facecolor='#eab308', edgecolor='white', label='Good Fit'),
            mpatches.Patch(facecolor='#f97316', edgecolor='white', label='System Dependent')
        ]
        ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-15, 140)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_ideal_xi_pitch.png", dpi=150, bbox_inches='tight', facecolor=pitch_color)
        plt.close()
        print("✓ Pitch visualization saved")
    
    def _create_recruitment_chart(self):
        """Create recruitment priorities chart"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df = self.recruitment_priorities
        colors = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308"}
        bar_colors = [colors.get(u, "#666") for u in df["urgency"]]
        
        y_pos = range(len(df))
        ax.barh(y_pos, df["cost_high"], color=bar_colors, alpha=0.7, edgecolor='white')
        ax.barh(y_pos, df["cost_low"], color=bar_colors, alpha=1.0, edgecolor='white')
        
        for i, row in df.iterrows():
            ax.text(row["cost_high"] + 2, i, f"£{row['cost_low']}M-£{row['cost_high']}M", va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["position"])
        ax.set_xlabel("Estimated Investment (£M)")
        ax.set_title("Recruitment Priorities", fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_recruitment_priorities.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Recruitment chart saved")
    
    def _create_summary_dashboard(self):
        """Create executive summary dashboard"""
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Aegis Analysis: {self.manager_dna['manager_name']} → Tottenham Hotspur",
                    fontsize=18, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # 1. DNA summary
        ax1 = fig.add_subplot(gs[0, 0])
        dims = list(self.dna_dimensions.items())[:6]
        y_pos = range(len(dims))
        ax1.barh(y_pos, [d[1] for d in dims], color='#1e88e5')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([d[0] for d in dims])
        ax1.set_xlim(0, 100)
        ax1.set_title("Manager DNA", fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Classification pie
        ax2 = fig.add_subplot(gs[0, 1])
        counts = self.fit_results["classification"].value_counts()
        colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
        ax2.pie(counts.values, labels=counts.index, colors=colors[:len(counts)], autopct='%1.0f%%', startangle=90)
        ax2.set_title("Squad Compatibility", fontweight='bold')
        
        # 3. Key metrics
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        avg_fit = self.fit_results["fit_score"].mean()
        key_enablers = len(self.fit_results[self.fit_results["classification"] == "Key Enabler"])
        total_value = self.squad_df["market_value"].sum() / 1e6
        total_investment = (self.recruitment_priorities["cost_low"].sum() + self.recruitment_priorities["cost_high"].sum()) / 2
        
        metrics_text = f"""
        KEY METRICS
        {'─' * 30}
        
        Average Fit Score: {avg_fit:.1f}
        Key Enablers: {key_enablers}
        Squad Value: £{total_value:.0f}M
        
        Critical Needs: {len(self.recruitment_priorities[self.recruitment_priorities['urgency']=='Critical'])}
        Est. Investment: £{total_investment:.0f}M
        
        Primary Formation: {self.manager_dna['formation_profile']['primary_formation']}
        Style: {self.manager_dna['tactical_profile']['playing_style'][:30]}
        """
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))
        
        # 4. Top fit players
        ax4 = fig.add_subplot(gs[1, 0])
        top5 = self.fit_results.head(5)
        y_pos = range(len(top5))
        colors = {"Key Enabler": "#22c55e", "Good Fit": "#eab308"}
        bar_colors = [colors.get(c, "#666") for c in top5["classification"]]
        ax4.barh(y_pos, top5["fit_score"], color=bar_colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top5["name"])
        ax4.set_xlim(0, 100)
        ax4.set_title("Top 5 Fit Players", fontweight='bold')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Recruitment priorities
        ax5 = fig.add_subplot(gs[1, 1])
        top_rec = self.recruitment_priorities.head(4)
        y_pos = range(len(top_rec))
        colors = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308"}
        bar_colors = [colors.get(u, "#666") for u in top_rec["urgency"]]
        ax5.barh(y_pos, top_rec["gap"], color=bar_colors)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top_rec["position"])
        ax5.set_title("Top Recruitment Needs", fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 6. Results summary
        ax6 = fig.add_subplot(gs[1, 2])
        results = self.manager_dna["results_profile"]
        categories = ['Wins', 'Draws', 'Losses']
        values = [results['wins'], results['draws'], results['losses']]
        ax6.pie(values, labels=categories, colors=['#22c55e', '#94a3b8', '#ef4444'], autopct='%1.0f%%', startangle=90)
        ax6.set_title(f"Results ({results['matches']} matches)", fontweight='bold')
        
        plt.savefig(OUTPUT_DIR / "07_executive_summary.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Executive summary saved")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    
    def save_outputs(self):
        """Save all analysis results"""
        
        print("\n" + "=" * 60)
        print("SAVING OUTPUTS")
        print("=" * 60)
        
        # Save CSVs
        self.fit_results.to_csv(OUTPUT_DIR / "squad_fit_scores.csv", index=False)
        self.ideal_xi.to_csv(OUTPUT_DIR / "ideal_xi.csv", index=False)
        self.recruitment_priorities.to_csv(OUTPUT_DIR / "recruitment_priorities.csv", index=False)
        
        # Save JSON summary
        summary = {
            "manager": self.manager_dna["manager_name"],
            "target_club": "Tottenham Hotspur",
            "matches_analyzed": self.manager_dna["matches_analyzed"],
            "primary_formation": self.manager_dna["formation_profile"]["primary_formation"],
            "playing_style": self.manager_dna["tactical_profile"]["playing_style"],
            "dna_dimensions": self.dna_dimensions,
            "squad_summary": {
                "total_players": len(self.fit_results),
                "average_fit": round(self.fit_results["fit_score"].mean(), 1),
                "key_enablers": len(self.fit_results[self.fit_results["classification"] == "Key Enabler"]),
                "good_fit": len(self.fit_results[self.fit_results["classification"] == "Good Fit"]),
            },
            "ideal_xi_avg_fit": round(self.ideal_xi["fit_score"].mean(), 1),
            "recruitment": {
                "priority_positions": self.recruitment_priorities["position"].tolist(),
                "total_investment_low": int(self.recruitment_priorities["cost_low"].sum()),
                "total_investment_high": int(self.recruitment_priorities["cost_high"].sum())
            }
        }
        
        with open(OUTPUT_DIR / "mtfi_analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved to: {OUTPUT_DIR}")
        print("  - squad_fit_scores.csv")
        print("  - ideal_xi.csv")
        print("  - recruitment_priorities.csv")
        print("  - mtfi_analysis_summary.json")
        print("  - 7 visualization PNGs")
        
        return self
    
    def run(self):
        """Execute complete Aegis analysis"""
        print("\n" + "#" * 60)
        print("# AEGIS Aegis COMPLETE ANALYSIS")
        print("#" * 60)
        
        return (self
                .load_data()
                .calculate_dna_dimensions()
                .calculate_squad_fit()
                .generate_ideal_xi()
                .analyze_recruitment()
                .create_all_visualizations()
                .save_outputs())


if __name__ == "__main__":
    AegisAnalyzer().run()
