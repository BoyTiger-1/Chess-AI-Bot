"""
Advanced Visualization Engine
Provides comprehensive visualization capabilities including interactive Plotly charts, D3.js visualizations,
3D plots, geospatial maps, network graphs, and dynamic dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import folium
from folium.plugins import HeatMap, MarkerCluster, Timeline
import altair as alt
import bokeh.plotting as bp
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool, PanTool, ResetTool, ZoomTool
from bokeh.layouts import column, row
from bokeh.io import show, output_notebook
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Chart types available."""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    NETWORK = "network"
    GEOGRAPHIC = "geographic"
    3D_SURFACE = "3d_surface"
    3D_SCATTER = "3d_scatter"
    RADAR = "radar"
    PARALLEL_COORDINATES = "parallel_coordinates"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    INDICATOR = "indicator"


class DashboardType(Enum):
    """Dashboard types."""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    REAL_TIME = "real_time"
    CUSTOM = "custom"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: str
    color_scheme: str = "default"
    width: int = 800
    height: int = 600
    theme: str = "plotly"
    animation: bool = False
    interactive: bool = True
    export_format: str = "html"
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardResult:
    """Dashboard creation result."""
    dashboard_id: str
    dashboard_type: DashboardType
    charts: List[Dict[str, Any]]
    layout_config: Dict[str, Any]
    filters: List[Dict[str, Any]]
    interactivity: Dict[str, Any]
    export_urls: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)


class VisualizationEngine:
    """
    Advanced Visualization Engine
    Provides comprehensive visualization capabilities with multiple chart types,
    interactive features, and dashboard creation.
    """
    
    def __init__(self, theme: str = "plotly"):
        """Initialize Visualization Engine."""
        self.theme = theme
        self.charts = {}
        self.dashboards = {}
        self.color_schemes = self._load_color_schemes()
        
    def create_interactive_line_chart(self,
                                    data: pd.DataFrame,
                                    x_column: str,
                                    y_columns: List[str],
                                    config: VisualizationConfig) -> Dict[str, Any]:
        """
        Create interactive line chart with multiple series.
        
        Args:
            data: DataFrame with data
            x_column: Column for x-axis
            y_columns: Columns for y-axes
            config: Visualization configuration
        """
        fig = go.Figure()
        
        colors = self._get_color_palette(len(y_columns))
        
        for i, y_column in enumerate(y_columns):
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='lines+markers',
                name=y_column,
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{y_column}</b><br>' +
                             f'{x_column}: %{{x}}<br>' +
                             f'{y_column}: %{{y}}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=x_column,
            yaxis_title=', '.join(y_columns),
            width=config.width,
            height=config.height,
            hovermode='x unified',
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add custom configuration
        fig.update_layout(**config.custom_config)
        
        # Convert to HTML if export is requested
        html_output = fig.to_html() if config.export_format == 'html' else None
        
        return {
            'chart_type': 'line_chart',
            'figure': fig,
            'html_output': html_output,
            'config': config,
            'data_summary': {
                'total_points': len(data),
                'series_count': len(y_columns),
                'date_range': (data[x_column].min(), data[x_column].max()) if pd.api.types.is_datetime64_any_dtype(data[x_column]) else None
            }
        }
    
    def create_interactive_bar_chart(self,
                                   data: pd.DataFrame,
                                   x_column: str,
                                   y_column: str,
                                   group_column: Optional[str] = None,
                                   config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create interactive bar chart with optional grouping.
        
        Args:
            data: DataFrame with data
            x_column: Column for x-axis
            y_column: Column for y-axis
            group_column: Optional grouping column
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.BAR,
                title=f"{y_column} by {x_column}",
                x_axis=x_column,
                y_axis=y_column
            )
        
        if group_column:
            # Grouped bar chart
            fig = px.bar(
                data,
                x=x_column,
                y=y_column,
                color=group_column,
                title=config.title,
                width=config.width,
                height=config.height
            )
        else:
            # Simple bar chart
            fig = px.bar(
                data,
                x=x_column,
                y=y_column,
                title=config.title,
                width=config.width,
                height=config.height
            )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            hovermode='x unified',
            showlegend=group_column is not None
        )
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': 'bar_chart',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config
        }
    
    def create_correlation_heatmap(self,
                                 data: pd.DataFrame,
                                 method: str = "pearson",
                                 config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create correlation heatmap for numeric data.
        
        Args:
            data: DataFrame with numeric columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.CORRELATION,
                title=f"Correlation Matrix ({method.title()})",
                x_axis="Variables",
                y_axis="Variables"
            )
        
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=self.theme,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': 'correlation_heatmap',
            'figure': fig,
            'correlation_matrix': corr_matrix.to_dict(),
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config
        }
    
    def create_network_graph(self,
                           edges: pd.DataFrame,
                           nodes: Optional[pd.DataFrame] = None,
                           layout: str = "spring",
                           config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create interactive network graph.
        
        Args:
            edges: DataFrame with source and target columns
            nodes: Optional DataFrame with node information
            layout: Graph layout algorithm
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.NETWORK,
                title="Network Graph",
                x_axis="",
                y_axis=""
            )
        
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(edges, source='source', target='target')
        
        # Add node attributes if provided
        if nodes is not None:
            for _, node in nodes.iterrows():
                if node['id'] in G.nodes():
                    for attr, value in node.items():
                        if attr != 'id':
                            G.nodes[node['id']][attr] = value
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract node and edge traces
        node_x, node_y = [], []
        node_text, node_color = [], []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_color.append(G.nodes[node].get('color', 'blue'))
            node_size.append(G.nodes[node].get('size', 10))
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=config.title,
                           template=self.theme,
                           width=config.width,
                           height=config.height,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network Graph",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': 'network_graph',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config,
            'graph_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G) if nx.is_connected(G) else 0
            }
        }
    
    def create_geographic_visualization(self,
                                      data: pd.DataFrame,
                                      lat_column: str,
                                      lon_column: str,
                                      value_column: str,
                                      map_type: str = "heatmap",
                                      config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create geographic visualization using Folium.
        
        Args:
            data: DataFrame with geographic coordinates
            lat_column: Latitude column name
            lon_column: Longitude column name
            value_column: Value column for visualization
            map_type: Type of map ('heatmap', 'markers', 'clusters')
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.GEOGRAPHIC,
                title="Geographic Visualization",
                x_axis="Longitude",
                y_axis="Latitude"
            )
        
        # Create base map
        center_lat = data[lat_column].mean()
        center_lon = data[lon_column].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        if map_type == "heatmap":
            # Heatmap
            heat_data = [[row[lat_column], row[lon_column], row[value_column]] 
                        for idx, row in data.iterrows()]
            HeatMap(heat_data).add_to(m)
            
        elif map_type == "markers":
            # Individual markers
            for idx, row in data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_column], row[lon_column]],
                    radius=row[value_column] / data[value_column].max() * 20,
                    popup=f"{value_column}: {row[value_column]}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
                
        elif map_type == "clusters":
            # Marker clusters
            marker_cluster = MarkerCluster().add_to(m)
            for idx, row in data.iterrows():
                folium.Marker(
                    location=[row[lat_column], row[lon_column]],
                    popup=f"{value_column}: {row[value_column]}"
                ).add_to(marker_cluster)
        
        # Add title
        title_html = f'''
                     <h3 align="center" style="font-size:20px"><b>{config.title}</b></h3>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Convert to HTML
        map_html = m._repr_html_()
        
        return {
            'chart_type': 'geographic_map',
            'figure': m,
            'html_output': map_html,
            'config': config,
            'map_stats': {
                'total_points': len(data),
                'coverage_area': {
                    'lat_range': (data[lat_column].min(), data[lat_column].max()),
                    'lon_range': (data[lon_column].min(), data[lon_column].max())
                }
            }
        }
    
    def create_3d_visualization(self,
                              data: pd.DataFrame,
                              x_column: str,
                              y_column: str,
                              z_column: str,
                              color_column: Optional[str] = None,
                              size_column: Optional[str] = None,
                              chart_type: str = "scatter",
                              config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create 3D visualization (scatter or surface).
        
        Args:
            data: DataFrame with 3D coordinates
            x_column, y_column, z_column: 3D coordinate columns
            color_column: Optional column for coloring
            size_column: Optional column for sizing
            chart_type: 'scatter' or 'surface'
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType["3D_" + chart_type.upper()],
                title=f"3D {chart_type.title()} Plot",
                x_axis=x_column,
                y_axis=y_column,
                z_axis=z_column
            )
        
        if chart_type == "scatter":
            # 3D scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=data[x_column],
                y=data[y_column],
                z=data[z_column],
                mode='markers',
                marker=dict(
                    size=data[size_column] if size_column else 5,
                    color=data[color_column] if color_column else data[z_column],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=data.index if not color_column else data[color_column],
                hovertemplate=f'{x_column}: %{{x}}<br>' +
                             f'{y_column}: %{{y}}<br>' +
                             f'{z_column}: %{{z}}<br>' +
                             '<extra></extra>'
            ))
            
        elif chart_type == "surface":
            # 3D surface plot
            # Create a grid for surface
            x_unique = sorted(data[x_column].unique())
            y_unique = sorted(data[y_column].unique())
            
            z_grid = []
            for y in y_unique:
                row = []
                for x in x_unique:
                    val = data[(data[x_column] == x) & (data[y_column] == y)][z_column]
                    row.append(val.iloc[0] if len(val) > 0 else 0)
                z_grid.append(row)
            
            fig = go.Figure(data=[go.Surface(
                x=x_unique,
                y=y_unique,
                z=z_grid,
                colorscale='Viridis'
            )])
        
        # Update layout
        fig.update_layout(
            title=config.title,
            scene=dict(
                xaxis_title=x_column,
                yaxis_title=y_column,
                zaxis_title=z_column,
                bgcolor='white'
            ),
            width=config.width,
            height=config.height,
            template=self.theme
        )
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': f'3d_{chart_type}',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config
        }
    
    def create_waterfall_chart(self,
                             data: pd.DataFrame,
                             category_column: str,
                             value_column: str,
                             config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create waterfall chart for showing cumulative effects.
        
        Args:
            data: DataFrame with category and value data
            category_column: Category column
            value_column: Value column
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.WATERFALL,
                title=f"Waterfall Chart - {value_column}",
                x_axis=category_column,
                y_axis=value_column
            )
        
        # Calculate cumulative values for waterfall
        cumulative = [0]
        for value in data[value_column]:
            cumulative.append(cumulative[-1] + value)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(data),
            x=data[category_column],
            y=data[value_column],
            text=[f"+{val}" if val > 0 else str(val) for val in data[value_column]],
            textposition="outside",
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"red"}},
            increasing={"marker":{"color":"green"}},
            totals={"marker":{"color":"blue"}}
        ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=category_column,
            yaxis_title=value_column,
            width=config.width,
            height=config.height,
            template=self.theme
        )
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': 'waterfall_chart',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config
        }
    
    def create_sankey_diagram(self,
                            source_data: pd.DataFrame,
                            source_column: str,
                            target_column: str,
                            value_column: str,
                            config: VisualizationConfig = None) -> Dict[str, Any]:
        """
        Create Sankey diagram for flow visualization.
        
        Args:
            source_data: DataFrame with source, target, and value columns
            source_column: Source node column
            target_column: Target node column
            value_column: Flow value column
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.SANKEY,
                title="Sankey Flow Diagram",
                x_axis="Source",
                y_axis="Target"
            )
        
        # Get unique nodes
        all_nodes = list(set(source_data[source_column]) | set(source_data[target_column]))
        
        # Create node mapping
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        # Create flow data
        source = [node_map[node] for node in source_data[source_column]]
        target = [node_map[node] for node in source_data[target_column]]
        values = source_data[value_column].tolist()
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color="rgba(0,100,80,0.4)"
            )
        )])
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=self.theme
        )
        
        fig.update_layout(**config.custom_config)
        
        return {
            'chart_type': 'sankey_diagram',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config,
            'flow_stats': {
                'total_flow': sum(values),
                'nodes': len(all_nodes),
                'flows': len(source_data)
            }
        }
    
    def create_dashboard(self,
                        charts: List[Dict[str, Any]],
                        layout_type: str = "grid",
                        filters: List[Dict[str, Any]] = None,
                        dashboard_type: DashboardType = DashboardType.CUSTOM) -> DashboardResult:
        """
        Create interactive dashboard with multiple charts.
        
        Args:
            charts: List of chart configurations and data
            layout_type: Dashboard layout ('grid', 'tabs', 'sidebar')
            filters: Optional dashboard filters
            dashboard_type: Type of dashboard
        """
        dashboard_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create layout configuration
        if layout_type == "grid":
            layout_config = self._create_grid_layout(charts)
        elif layout_type == "tabs":
            layout_config = self._create_tabs_layout(charts)
        elif layout_type == "sidebar":
            layout_config = self._create_sidebar_layout(charts)
        else:
            layout_config = self._create_grid_layout(charts)
        
        # Add interactivity
        interactivity = {
            'cross_filtering': True,
            'drill_down': True,
            'export_capabilities': ['png', 'pdf', 'html'],
            'real_time_updates': dashboard_type == DashboardType.REAL_TIME
        }
        
        # Create HTML dashboard
        dashboard_html = self._generate_dashboard_html(charts, layout_config, filters, dashboard_type)
        
        # Generate export URLs
        export_urls = {
            'html': f"/dashboards/{dashboard_id}/export/html",
            'pdf': f"/dashboards/{dashboard_id}/export/pdf",
            'png': f"/dashboards/{dashboard_id}/export/png"
        }
        
        # Store dashboard
        dashboard = DashboardResult(
            dashboard_id=dashboard_id,
            dashboard_type=dashboard_type,
            charts=charts,
            layout_config=layout_config,
            filters=filters or [],
            interactivity=interactivity,
            export_urls=export_urls
        )
        
        self.dashboards[dashboard_id] = dashboard
        
        return dashboard
    
    def create_executive_dashboard(self, business_metrics: Dict[str, Any]) -> DashboardResult:
        """
        Create executive dashboard with key business metrics.
        
        Args:
            business_metrics: Dictionary containing key business metrics
        """
        charts = []
        
        # Revenue trend chart
        if 'revenue_data' in business_metrics:
            revenue_chart = self.create_interactive_line_chart(
                business_metrics['revenue_data'],
                'date',
                ['revenue'],
                VisualizationConfig(
                    chart_type=ChartType.LINE,
                    title="Revenue Trend",
                    width=600,
                    height=400
                )
            )
            charts.append(revenue_chart)
        
        # Customer acquisition chart
        if 'customer_data' in business_metrics:
            customer_chart = self.create_interactive_bar_chart(
                business_metrics['customer_data'],
                'month',
                'new_customers',
                config=VisualizationConfig(
                    chart_type=ChartType.BAR,
                    title="New Customer Acquisition",
                    width=400,
                    height=300
                )
            )
            charts.append(customer_chart)
        
        # KPI indicators
        if 'kpis' in business_metrics:
            kpi_charts = self._create_kpi_indicators(business_metrics['kpis'])
            charts.extend(kpi_charts)
        
        # Executive layout
        layout_config = {
            'type': 'executive',
            'rows': 3,
            'columns': 3,
            'arrangement': 'metrics_focused'
        }
        
        filters = [
            {'type': 'date_range', 'field': 'date', 'label': 'Time Period'},
            {'type': 'select', 'field': 'region', 'label': 'Region', 'options': ['All', 'North America', 'Europe', 'Asia']},
            {'type': 'multi_select', 'field': 'product', 'label': 'Product Line', 'options': ['All Products']}
        ]
        
        return self.create_dashboard(
            charts=charts,
            layout_type="grid",
            filters=filters,
            dashboard_type=DashboardType.EXECUTIVE
        )
    
    def create_operational_dashboard(self, operational_data: Dict[str, Any]) -> DashboardResult:
        """
        Create operational dashboard with real-time monitoring.
        
        Args:
            operational_data: Dictionary containing operational metrics
        """
        charts = []
        
        # System performance metrics
        if 'performance_data' in operational_data:
            performance_chart = self.create_line_chart(
                operational_data['performance_data'],
                'timestamp',
                ['cpu_usage', 'memory_usage', 'disk_usage'],
                VisualizationConfig(
                    chart_type=ChartType.LINE,
                    title="System Performance",
                    width=800,
                    height=400,
                    animation=True
                )
            )
            charts.append(performance_chart)
        
        # Error rates and alerts
        if 'alerts_data' in operational_data:
            alerts_chart = self.create_area_chart(
                operational_data['alerts_data'],
                'time',
                ['critical', 'warning', 'info'],
                VisualizationConfig(
                    chart_type=ChartType.AREA,
                    title="Alert Trends",
                    width=600,
                    height=300
                )
            )
            charts.append(alerts_chart)
        
        # Real-time indicators
        if 'real_time_metrics' in operational_data:
            realtime_charts = self._create_realtime_indicators(operational_data['real_time_metrics'])
            charts.extend(realtime_charts)
        
        # Operational layout
        layout_config = {
            'type': 'operational',
            'rows': 4,
            'columns': 2,
            'arrangement': 'monitoring_focused'
        }
        
        filters = [
            {'type': 'time_window', 'field': 'timestamp', 'label': 'Time Window', 'options': ['1h', '6h', '24h', '7d']},
            {'type': 'severity', 'field': 'level', 'label': 'Alert Level', 'options': ['All', 'Critical', 'Warning', 'Info']}
        ]
        
        return self.create_dashboard(
            charts=charts,
            layout_type="grid",
            filters=filters,
            dashboard_type=DashboardType.OPERATIONAL
        )
    
    def export_visualization(self,
                           chart_data: Dict[str, Any],
                           format: str = "png",
                           filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export visualization to different formats.
        
        Args:
            chart_data: Chart data dictionary
            format: Export format ('png', 'pdf', 'html', 'svg')
            filename: Output filename
        """
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if format == "html":
                # HTML export (already available)
                export_path = f"/exports/{filename}.html"
                with open(export_path, 'w') as f:
                    f.write(chart_data.get('html_output', ''))
                    
            elif format == "png":
                # PNG export using Kaleido
                try:
                    import plotly.io as pio
                    fig = chart_data['figure']
                    pio.write_image(fig, f"/exports/{filename}.png", format='png', width=1200, height=800)
                    export_path = f"/exports/{filename}.png"
                except ImportError:
                    export_path = f"/exports/{filename}.html"  # Fallback to HTML
                    
            elif format == "pdf":
                # PDF export
                fig = chart_data['figure']
                fig.write_image(f"/exports/{filename}.pdf", format='pdf')
                export_path = f"/exports/{filename}.pdf"
                
            elif format == "svg":
                # SVG export
                fig = chart_data['figure']
                fig.write_image(f"/exports/{filename}.svg", format='svg')
                export_path = f"/exports/{filename}.svg"
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return {
                'success': True,
                'export_path': export_path,
                'filename': filename,
                'format': format
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_custom_visualization(self,
                                  data: pd.DataFrame,
                                  visualization_type: str,
                                  custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create custom visualization with user-defined parameters.
        
        Args:
            data: DataFrame with data
            visualization_type: Type of visualization
            custom_config: Custom configuration parameters
        """
        # This would handle various custom visualization requirements
        # For now, return a placeholder
        
        return {
            'chart_type': 'custom',
            'visualization_type': visualization_type,
            'config': custom_config,
            'data_summary': {
                'rows': len(data),
                'columns': len(data.columns),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns)
            },
            'message': f"Custom {visualization_type} visualization created with provided configuration"
        }
    
    # Helper methods
    
    def _load_color_schemes(self) -> Dict[str, List[str]]:
        """Load predefined color schemes."""
        return {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'pastel': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'],
            'bright': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'],
            'dark': ['#2d3436', '#636e72', '#00b894', '#0984e3', '#fdcb6e', '#e17055'],
            'professional': ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1']
        }
    
    def _get_color_palette(self, n_colors: int) -> List[str]:
        """Get color palette for given number of colors."""
        if n_colors <= len(self.color_schemes['default']):
            return self.color_schemes['default'][:n_colors]
        else:
            # Repeat colors if more needed
            colors = self.color_schemes['default']
            while len(colors) < n_colors:
                colors.extend(self.color_schemes['default'])
            return colors[:n_colors]
    
    def _create_grid_layout(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create grid layout configuration."""
        n_charts = len(charts)
        if n_charts <= 2:
            rows, cols = 1, n_charts
        elif n_charts <= 4:
            rows, cols = 2, 2
        elif n_charts <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        return {
            'type': 'grid',
            'rows': rows,
            'columns': cols,
            'chart_positions': [(i // cols, i % cols) for i in range(n_charts)]
        }
    
    def _create_tabs_layout(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create tabs layout configuration."""
        return {
            'type': 'tabs',
            'tabs': [{'title': f'Chart {i+1}', 'chart_index': i} for i in range(len(charts))]
        }
    
    def _create_sidebar_layout(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create sidebar layout configuration."""
        return {
            'type': 'sidebar',
            'sidebar_width': '25%',
            'main_content_width': '75%',
            'sidebar_charts': charts[:len(charts)//2],
            'main_charts': charts[len(charts)//2:]
        }
    
    def _generate_dashboard_html(self,
                               charts: List[Dict[str, Any]],
                               layout_config: Dict[str, Any],
                               filters: List[Dict[str, Any]],
                               dashboard_type: DashboardType) -> str:
        """Generate HTML for dashboard."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Intelligence Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .dashboard-header {{ text-align: center; margin-bottom: 20px; }}
                .chart-container {{ margin: 10px; }}
                .filters {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; }}
                .grid-container {{ display: grid; gap: 20px; }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>Business Intelligence Dashboard</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="filters">
                <h3>Filters</h3>
                {filters_content}
            </div>
            
            <div class="grid-container">
                {charts_content}
            </div>
        </body>
        </html>
        """
        
        # Generate filters content
        filters_content = "".join([f"<p>{filter_def['label']}: {filter_def['type']}</p>" 
                                 for filter_def in filters])
        
        # Generate charts content
        charts_content = ""
        for chart in charts:
            charts_content += f'<div class="chart-container">{chart.get("html_output", "Chart HTML")}</div>'
        
        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            filters_content=filters_content,
            charts_content=charts_content
        )
    
    def _create_kpi_indicators(self, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create KPI indicator charts."""
        kpi_charts = []
        
        for kpi_name, kpi_value in kpis.items():
            config = VisualizationConfig(
                chart_type=ChartType.INDICATOR,
                title=f"{kpi_name.replace('_', ' ').title()}",
                x_axis="",
                y_axis="",
                width=300,
                height=200
            )
            
            # Create gauge/indicator
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=kpi_value,
                domain={{'x': [0, 1], 'y': [0, 1]}},
                title={{'text': kpi_name.replace('_', ' ').title()}},
                delta={{'reference': kpi_value * 0.9}},
                gauge={{
                    'axis': {{'range': [None, kpi_value * 1.5]}},
                    'bar': {{'color': "darkblue"}},
                    'steps': [
                        {{'range': [0, kpi_value * 0.5], 'color': "lightgray"}},
                        {{'range': [kpi_value * 0.5, kpi_value], 'color': "gray"}}
                    ],
                    'threshold': {{
                        'line': {{'color': "red", 'width': 4}},
                        'thickness': 0.75,
                        'value': kpi_value * 1.2
                    }}
                }}
            ))
            
            kpi_charts.append({
                'chart_type': 'kpi_indicator',
                'figure': fig,
                'html_output': fig.to_html(),
                'config': config
            })
        
        return kpi_charts
    
    def _create_realtime_indicators(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create real-time monitoring indicators."""
        # Similar to KPI indicators but with auto-refresh capability
        return self._create_kpi_indicators(metrics)
    
    def line_chart(self, data: pd.DataFrame, x_column: str, y_columns: List[str], config: VisualizationConfig) -> Dict[str, Any]:
        """Simplified line chart method."""
        return self.create_interactive_line_chart(data, x_column, y_columns, config)
    
    def area_chart(self, data: pd.DataFrame, x_column: str, y_columns: List[str], config: VisualizationConfig) -> Dict[str, Any]:
        """Simplified area chart method."""
        if config is None:
            config = VisualizationConfig(
                chart_type=ChartType.AREA,
                title=f"Area Chart - {', '.join(y_columns)}",
                x_axis=x_column,
                y_axis=", ".join(y_columns)
            )
        
        fig = go.Figure()
        
        colors = self._get_color_palette(len(y_columns))
        
        for i, y_column in enumerate(y_columns):
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                name=y_column,
                line=dict(color=colors[i])
            ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=x_column,
            yaxis_title=", ".join(y_columns),
            width=config.width,
            height=config.height,
            template=self.theme
        )
        
        return {
            'chart_type': 'area_chart',
            'figure': fig,
            'html_output': fig.to_html() if config.export_format == 'html' else None,
            'config': config
        }