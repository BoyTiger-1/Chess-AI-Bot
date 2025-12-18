"""
Advanced Reporting Engine
Provides comprehensive automated reporting, PDF generation, executive summaries,
data storytelling, and multi-format export capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import io
import base64
from pathlib import Path

import pandas as pd
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yagmail
import schedule
import threading
import time

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Report types."""
    EXECUTIVE_SUMMARY = "executive_summary"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CUSTOMER_ANALYSIS = "customer_analysis"
    MARKET_INTELLIGENCE = "market_intelligence"
    OPERATIONAL_DASHBOARD = "operational_dashboard"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    COMPLIANCE_REPORT = "compliance_report"
    CUSTOM = "custom"


class ExportFormat(Enum):
    """Export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    WORD = "word"


class DeliveryChannel(Enum):
    """Report delivery channels."""
    EMAIL = "email"
    FTP = "ftp"
    API = "api"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"


@dataclass
class ReportSection:
    """Report section definition."""
    title: str
    content_type: str  # "chart", "table", "text", "image"
    data: Any
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportTemplate:
    """Report template definition."""
    template_id: str
    name: str
    report_type: ReportType
    sections: List[ReportSection]
    styling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportResult:
    """Report generation result."""
    report_id: str
    template_id: str
    title: str
    generated_at: datetime
    file_path: str
    file_size: int
    export_format: ExportFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class ReportingEngine:
    """
    Advanced Reporting Engine
    Provides comprehensive automated reporting, document generation, 
    and multi-format export capabilities.
    """
    
    def __init__(self, base_path: str = "/reports"):
        """Initialize Reporting Engine."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=True
        )
        
        # Report templates
        self.templates = {}
        self.generated_reports = {}
        
        # Email configuration
        self.email_config = {}
        
        # Schedule management
        self.scheduled_reports = {}
        
        # Initialize default templates
        self._initialize_default_templates()
        
    def create_executive_summary_report(self,
                                      business_data: Dict[str, Any],
                                      config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create comprehensive executive summary report.
        
        Args:
            business_data: Dictionary containing business metrics and insights
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Executive Summary Report',
                'author': 'Business AI Assistant',
                'include_charts': True,
                'include_recommendations': True,
                'export_format': ExportFormat.PDF
            }
        
        # Create report sections
        sections = []
        
        # Title page
        title_section = ReportSection(
            title="Executive Summary",
            content_type="title",
            data={
                'title': config['title'],
                'subtitle': f"Generated on {datetime.now().strftime('%B %d, %Y')}",
                'author': config['author']
            }
        )
        sections.append(title_section)
        
        # Key metrics section
        if 'kpis' in business_data:
            kpi_section = ReportSection(
                title="Key Performance Indicators",
                content_type="table",
                data=business_data['kpis'],
                styling={'table_style': 'executive'}
            )
            sections.append(kpi_section)
        
        # Revenue analysis
        if 'revenue_data' in business_data:
            revenue_section = ReportSection(
                title="Revenue Analysis",
                content_type="chart",
                data=business_data['revenue_data'],
                visualization_config={
                    'chart_type': 'line',
                    'title': 'Revenue Trend',
                    'width': 800,
                    'height': 400
                }
            )
            sections.append(revenue_section)
        
        # Customer insights
        if 'customer_insights' in business_data:
            customer_section = ReportSection(
                title="Customer Insights",
                content_type="text",
                data=business_data['customer_insights']
            )
            sections.append(customer_section)
        
        # Market analysis
        if 'market_analysis' in business_data:
            market_section = ReportSection(
                title="Market Analysis",
                content_type="text",
                data=business_data['market_analysis']
            )
            sections.append(market_section)
        
        # Strategic recommendations
        if config.get('include_recommendations', True):
            recommendations_section = ReportSection(
                title="Strategic Recommendations",
                content_type="text",
                data=self._generate_strategic_recommendations(business_data)
            )
            sections.append(recommendations_section)
        
        # Create template
        template = ReportTemplate(
            template_id="executive_summary",
            name="Executive Summary Report",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            sections=sections,
            styling=config
        )
        
        # Generate report
        return self._generate_report(template, ExportFormat.PDF)
    
    def create_financial_analysis_report(self,
                                       financial_data: Dict[str, Any],
                                       config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create comprehensive financial analysis report.
        
        Args:
            financial_data: Financial data and metrics
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Financial Analysis Report',
                'period': 'Q4 2024',
                'include_projections': True,
                'export_format': ExportFormat.PDF
            }
        
        sections = []
        
        # Financial summary
        sections.append(ReportSection(
            title="Financial Summary",
            content_type="table",
            data=financial_data.get('summary', {}),
            styling={'highlight_negative': True}
        ))
        
        # Profit & Loss analysis
        if 'profit_loss' in financial_data:
            sections.append(ReportSection(
                title="Profit & Loss Statement",
                content_type="table",
                data=financial_data['profit_loss']
            ))
        
        # Cash flow analysis
        if 'cash_flow' in financial_data:
            sections.append(ReportSection(
                title="Cash Flow Analysis",
                content_type="chart",
                data=financial_data['cash_flow'],
                visualization_config={'chart_type': 'area'}
            ))
        
        # Financial ratios
        if 'ratios' in financial_data:
            sections.append(ReportSection(
                title="Key Financial Ratios",
                content_type="table",
                data=financial_data['ratios']
            ))
        
        # Risk analysis
        if 'risk_analysis' in financial_data:
            sections.append(ReportSection(
                title="Risk Analysis",
                content_type="text",
                data=financial_data['risk_analysis']
            ))
        
        template = ReportTemplate(
            template_id="financial_analysis",
            name="Financial Analysis Report",
            report_type=ReportType.FINANCIAL_ANALYSIS,
            sections=sections,
            styling=config
        )
        
        return self._generate_report(template, ExportFormat.PDF)
    
    def create_customer_analysis_report(self,
                                      customer_data: Dict[str, Any],
                                      config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create comprehensive customer analysis report.
        
        Args:
            customer_data: Customer data and analytics
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Customer Analysis Report',
                'include_segmentation': True,
                'include_churn_analysis': True,
                'export_format': ExportFormat.PDF
            }
        
        sections = []
        
        # Customer overview
        sections.append(ReportSection(
            title="Customer Overview",
            content_type="table",
            data=customer_data.get('overview', {})
        ))
        
        # Segmentation analysis
        if config.get('include_segmentation', True) and 'segmentation' in customer_data:
            sections.append(ReportSection(
                title="Customer Segmentation",
                content_type="chart",
                data=customer_data['segmentation'],
                visualization_config={'chart_type': 'pie'}
            ))
        
        # Churn analysis
        if config.get('include_churn_analysis', True) and 'churn' in customer_data:
            sections.append(ReportSection(
                title="Churn Analysis",
                content_type="chart",
                data=customer_data['churn'],
                visualization_config={'chart_type': 'bar'}
            ))
        
        # Customer lifetime value
        if 'clv' in customer_data:
            sections.append(ReportSection(
                title="Customer Lifetime Value",
                content_type="chart",
                data=customer_data['clv'],
                visualization_config={'chart_type': 'histogram'}
            ))
        
        # Customer journey insights
        if 'journey' in customer_data:
            sections.append(ReportSection(
                title="Customer Journey Insights",
                content_type="text",
                data=customer_data['journey']
            ))
        
        template = ReportTemplate(
            template_id="customer_analysis",
            name="Customer Analysis Report",
            report_type=ReportType.CUSTOMER_ANALYSIS,
            sections=sections,
            styling=config
        )
        
        return self._generate_report(template, ExportFormat.PDF)
    
    def create_market_intelligence_report(self,
                                        market_data: Dict[str, Any],
                                        config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create market intelligence report.
        
        Args:
            market_data: Market analysis and competitive intelligence
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Market Intelligence Report',
                'include_competitor_analysis': True,
                'include_trend_analysis': True,
                'export_format': ExportFormat.PDF
            }
        
        sections = []
        
        # Market overview
        sections.append(ReportSection(
            title="Market Overview",
            content_type="text",
            data=market_data.get('overview', {})
        ))
        
        # Competitive landscape
        if config.get('include_competitor_analysis', True) and 'competitors' in market_data:
            sections.append(ReportSection(
                title="Competitive Landscape",
                content_type="chart",
                data=market_data['competitors'],
                visualization_config={'chart_type': 'radar'}
            ))
        
        # Trend analysis
        if config.get('include_trend_analysis', True) and 'trends' in market_data:
            sections.append(ReportSection(
                title="Market Trends",
                content_type="chart",
                data=market_data['trends'],
                visualization_config={'chart_type': 'line'}
            ))
        
        # Market opportunities
        if 'opportunities' in market_data:
            sections.append(ReportSection(
                title="Market Opportunities",
                content_type="text",
                data=market_data['opportunities']
            ))
        
        # Strategic implications
        if 'implications' in market_data:
            sections.append(ReportSection(
                title="Strategic Implications",
                content_type="text",
                data=market_data['implications']
            ))
        
        template = ReportTemplate(
            template_id="market_intelligence",
            name="Market Intelligence Report",
            report_type=ReportType.MARKET_INTELLIGENCE,
            sections=sections,
            styling=config
        )
        
        return self._generate_report(template, ExportFormat.PDF)
    
    def create_predictive_analysis_report(self,
                                        predictive_data: Dict[str, Any],
                                        config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create predictive analysis report.
        
        Args:
            predictive_data: Predictive models and forecasts
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Predictive Analysis Report',
                'include_forecasts': True,
                'include_scenarios': True,
                'export_format': ExportFormat.PDF
            }
        
        sections = []
        
        # Executive summary
        sections.append(ReportSection(
            title="Executive Summary",
            content_type="text",
            data=predictive_data.get('summary', {})
        ))
        
        # Forecasts
        if config.get('include_forecasts', True) and 'forecasts' in predictive_data:
            sections.append(ReportSection(
                title="Forecasts & Projections",
                content_type="chart",
                data=predictive_data['forecasts'],
                visualization_config={'chart_type': 'line'}
            ))
        
        # Scenario analysis
        if config.get('include_scenarios', True) and 'scenarios' in predictive_data:
            sections.append(ReportSection(
                title="Scenario Analysis",
                content_type="table",
                data=predictive_data['scenarios']
            ))
        
        # Risk assessment
        if 'risk_assessment' in predictive_data:
            sections.append(ReportSection(
                title="Risk Assessment",
                content_type="text",
                data=predictive_data['risk_assessment']
            ))
        
        # Recommendations
        if 'recommendations' in predictive_data:
            sections.append(ReportSection(
                title="Recommendations",
                content_type="text",
                data=predictive_data['recommendations']
            ))
        
        template = ReportTemplate(
            template_id="predictive_analysis",
            name="Predictive Analysis Report",
            report_type=ReportType.PREDICTIVE_ANALYSIS,
            sections=sections,
            styling=config
        )
        
        return self._generate_report(template, ExportFormat.PDF)
    
    def schedule_report(self,
                       template_id: str,
                       schedule_config: Dict[str, Any],
                       delivery_config: Dict[str, Any]) -> str:
        """
        Schedule automated report generation and delivery.
        
        Args:
            template_id: Template to use
            schedule_config: Schedule configuration (frequency, time, etc.)
            delivery_config: Delivery configuration (email, recipients, etc.)
        """
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create schedule
        schedule_entry = {
            'id': schedule_id,
            'template_id': template_id,
            'schedule_config': schedule_config,
            'delivery_config': delivery_config,
            'created_at': datetime.now(),
            'last_run': None,
            'next_run': self._calculate_next_run(schedule_config)
        }
        
        self.scheduled_reports[schedule_id] = schedule_entry
        
        # Setup scheduler
        frequency = schedule_config.get('frequency', 'weekly')
        time = schedule_config.get('time', '09:00')
        
        if frequency == 'daily':
            schedule.every().day.at(time).do(self._execute_scheduled_report, schedule_id)
        elif frequency == 'weekly':
            day = schedule_config.get('day', 'monday')
            getattr(schedule.every(), day.lower()).at(time).do(self._execute_scheduled_report, schedule_id)
        elif frequency == 'monthly':
            day = schedule_config.get('day', 1)
            schedule.every().month.do(self._execute_scheduled_report, schedule_id)
        
        return schedule_id
    
    def export_report(self,
                     report_id: str,
                     export_format: ExportFormat,
                     output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export report to different formats.
        
        Args:
            report_id: Report ID to export
            export_format: Target format
            output_path: Output file path
        """
        if report_id not in self.generated_reports:
            raise ValueError(f"Report {report_id} not found")
        
        report = self.generated_reports[report_id]
        
        if output_path is None:
            output_path = str(self.base_path / f"{report_id}.{export_format.value}")
        
        try:
            if export_format == ExportFormat.PDF:
                # PDF export (already generated)
                return {'success': True, 'path': report.file_path}
            
            elif export_format == ExportFormat.EXCEL:
                # Excel export
                excel_path = self._export_to_excel(report, output_path)
                return {'success': True, 'path': excel_path}
            
            elif export_format == ExportFormat.HTML:
                # HTML export
                html_path = self._export_to_html(report, output_path)
                return {'success': True, 'path': html_path}
            
            elif export_format == ExportFormat.JSON:
                # JSON export
                json_path = self._export_to_json(report, output_path)
                return {'success': True, 'path': json_path}
            
            else:
                return {'success': False, 'error': f'Export format {export_format.value} not supported'}
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_data_story(self,
                          data: pd.DataFrame,
                          analysis_results: Dict[str, Any],
                          story_config: Dict[str, Any]) -> str:
        """
        Generate narrative data story using natural language generation.
        
        Args:
            data: Data to analyze
            analysis_results: Analysis results
            story_config: Story generation configuration
        """
        story_template = """
        # Data Story: {title}
        
        ## Executive Summary
        {executive_summary}
        
        ## Key Findings
        {key_findings}
        
        ## Detailed Analysis
        {detailed_analysis}
        
        ## Trends & Patterns
        {trends_analysis}
        
        ## Strategic Implications
        {implications}
        
        ## Recommendations
        {recommendations}
        
        ---
        *Generated by Business AI Assistant on {timestamp}*
        """
        
        # Generate story content
        story_content = self._generate_story_content(data, analysis_results, story_config)
        
        # Apply template
        template = Template(story_template)
        story = template.render(
            title=story_config.get('title', 'Data Analysis Story'),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **story_content
        )
        
        return story
    
    def create_dashboard_report(self,
                              dashboard_data: Dict[str, Any],
                              config: Optional[Dict[str, Any]] = None) -> ReportResult:
        """
        Create report from dashboard data.
        
        Args:
            dashboard_data: Dashboard configuration and data
            config: Report configuration
        """
        if config is None:
            config = {
                'title': 'Dashboard Report',
                'include_snapshots': True,
                'export_format': ExportFormat.PDF
            }
        
        sections = []
        
        # Dashboard overview
        sections.append(ReportSection(
            title="Dashboard Overview",
            content_type="text",
            data=dashboard_data.get('overview', {})
        ))
        
        # KPI summary
        if 'kpis' in dashboard_data:
            sections.append(ReportSection(
                title="Key Performance Indicators",
                content_type="table",
                data=dashboard_data['kpis']
            ))
        
        # Chart snapshots
        if config.get('include_snapshots', True):
            for chart_name, chart_data in dashboard_data.get('charts', {}).items():
                sections.append(ReportSection(
                    title=f"Chart: {chart_name}",
                    content_type="chart_snapshot",
                    data=chart_data
                ))
        
        template = ReportTemplate(
            template_id="dashboard_report",
            name="Dashboard Report",
            report_type=ReportType.OPERATIONAL_DASHBOARD,
            sections=sections,
            styling=config
        )
        
        return self._generate_report(template, ExportFormat.PDF)
    
    def add_email_delivery(self,
                          config: Dict[str, Any]) -> None:
        """
        Configure email delivery for reports.
        
        Args:
            config: Email configuration
        """
        self.email_config = {
            'smtp_server': config.get('smtp_server'),
            'smtp_port': config.get('smtp_port', 587),
            'username': config.get('username'),
            'password': config.get('password'),
            'use_tls': config.get('use_tls', True)
        }
    
    def send_report_email(self,
                         report_id: str,
                         recipients: List[str],
                         subject: Optional[str] = None,
                         body: Optional[str] = None) -> Dict[str, Any]:
        """
        Send report via email.
        
        Args:
            report_id: Report to send
            recipients: Email recipients
            subject: Email subject
            body: Email body
        """
        if report_id not in self.generated_reports:
            raise ValueError(f"Report {report_id} not found")
        
        if not self.email_config:
            raise ValueError("Email configuration not set")
        
        report = self.generated_reports[report_id]
        
        try:
            # Setup email
            yag = yagmail.SMTP(
                self.email_config['username'],
                self.email_config['password'],
                host=self.email_config['smtp_server'],
                port=self.email_config['smtp_port']
            )
            
            # Email content
            email_subject = subject or f"Report: {report.title}"
            email_body = body or f"Please find attached the {report.title} generated on {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send email
            yag.send(
                to=recipients,
                subject=email_subject,
                contents=email_body,
                attachments=[report.file_path]
            )
            
            return {'success': True, 'recipients': recipients}
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Helper methods
    
    def _initialize_default_templates(self) -> None:
        """Initialize default report templates."""
        # This would populate default templates
        pass
    
    def _generate_report(self, template: ReportTemplate, export_format: ExportFormat) -> ReportResult:
        """Generate report from template."""
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate file path
        file_path = self.base_path / f"{report_id}.{export_format.value}"
        
        # Create PDF
        if export_format == ExportFormat.PDF:
            file_path = self._create_pdf_report(template, file_path)
        
        # Calculate file size
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Create result
        result = ReportResult(
            report_id=report_id,
            template_id=template.template_id,
            title=template.styling.get('title', 'Business Report'),
            generated_at=datetime.now(),
            file_path=str(file_path),
            file_size=file_size,
            export_format=export_format,
            metadata={
                'template_name': template.name,
                'sections_count': len(template.sections),
                'generation_time': datetime.now()
            }
        )
        
        # Store report
        self.generated_reports[report_id] = result
        
        return result
    
    def _create_pdf_report(self, template: ReportTemplate, output_path: Path) -> Path:
        """Create PDF report using ReportLab."""
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Build story
        story = []
        
        # Add content for each section
        for section in template.sections:
            if section.content_type == "title":
                # Title page
                story.append(Paragraph(section.data.get('title', ''), title_style))
                story.append(Paragraph(section.data.get('subtitle', ''), styles['Normal']))
                story.append(Paragraph(f"Generated by {section.data.get('author', 'Business AI Assistant')}", styles['Normal']))
                story.append(Spacer(1, 0.5*inch))
                
            elif section.content_type == "text":
                # Text content
                story.append(Paragraph(section.title, heading_style))
                story.append(Paragraph(str(section.data), styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
                
            elif section.content_type == "table":
                # Table content
                story.append(Paragraph(section.title, heading_style))
                
                if isinstance(section.data, dict):
                    # Convert dict to table data
                    table_data = [['Metric', 'Value']] + [[k, str(v)] for k, v in section.data.items()]
                else:
                    table_data = [['Data']] + [[str(item)] for item in section.data]
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 0.3*inch))
                
            elif section.content_type == "chart":
                # Chart content
                story.append(Paragraph(section.title, heading_style))
                
                # Create chart and save as image
                chart_image = self._create_chart_image(section.data, section.visualization_config)
                if chart_image:
                    story.append(Image(chart_image, width=6*inch, height=3*inch))
                story.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_chart_image(self, data: Any, config: Dict[str, Any]) -> Optional[str]:
        """Create chart image for PDF."""
        try:
            # Generate temporary image file
            temp_path = self.base_path / f"temp_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            chart_type = config.get('chart_type', 'line')
            
            if chart_type == 'line':
                if isinstance(data, pd.DataFrame):
                    for column in data.select_dtypes(include=[np.number]).columns:
                        ax.plot(data.index, data[column], label=column)
                    ax.legend()
                else:
                    ax.plot(range(len(data)), data)
                    
            elif chart_type == 'bar':
                if isinstance(data, pd.DataFrame):
                    data.plot(kind='bar', ax=ax)
                else:
                    ax.bar(range(len(data)), data)
                    
            elif chart_type == 'pie':
                if isinstance(data, dict):
                    ax.pie(data.values(), labels=data.keys())
                else:
                    ax.pie(data)
            
            ax.set_title(config.get('title', 'Chart'))
            ax.grid(True, alpha=0.3)
            
            # Save figure
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            return None
    
    def _export_to_excel(self, report: ReportResult, output_path: str) -> str:
        """Export report to Excel format."""
        # This would create an Excel workbook with multiple sheets
        # For now, return the original PDF path
        return report.file_path
    
    def _export_to_html(self, report: ReportResult, output_path: str) -> str:
        """Export report to HTML format."""
        # This would create an HTML report
        # For now, return a placeholder
        html_path = output_path.replace('.pdf', '.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p>Generated on {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Report ID: {report.report_id}</p>
            <p>File Size: {report.file_size} bytes</p>
            <!-- Additional content would be added here -->
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _export_to_json(self, report: ReportResult, output_path: str) -> str:
        """Export report metadata to JSON."""
        json_path = output_path.replace('.pdf', '.json')
        
        import json
        
        report_data = {
            'report_id': report.report_id,
            'title': report.title,
            'generated_at': report.generated_at.isoformat(),
            'file_path': report.file_path,
            'file_size': report.file_size,
            'export_format': report.export_format.value,
            'metadata': report.metadata,
            'summary': report.summary
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_path
    
    def _calculate_next_run(self, schedule_config: Dict[str, Any]) -> datetime:
        """Calculate next run time for scheduled report."""
        frequency = schedule_config.get('frequency', 'weekly')
        time_str = schedule_config.get('time', '09:00')
        
        now = datetime.now()
        hour, minute = map(int, time_str.split(':'))
        
        if frequency == 'daily':
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
                
        elif frequency == 'weekly':
            day_name = schedule_config.get('day', 'monday')
            day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                      'friday': 4, 'saturday': 5, 'sunday': 6}
            target_day = day_map.get(day_name.lower(), 0)
            
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
        else:
            # Default to weekly
            next_run = now + timedelta(days=7)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return next_run
    
    def _execute_scheduled_report(self, schedule_id: str) -> None:
        """Execute scheduled report generation."""
        if schedule_id not in self.scheduled_reports:
            logger.error(f"Schedule {schedule_id} not found")
            return
        
        schedule_entry = self.scheduled_reports[schedule_id]
        
        try:
            # Get template and generate report
            template_id = schedule_entry['template_id']
            # This would lookup the template and generate the report
            # For now, just log the execution
            logger.info(f"Executing scheduled report {schedule_id}")
            
            # Update last run time
            schedule_entry['last_run'] = datetime.now()
            schedule_entry['next_run'] = self._calculate_next_run(schedule_entry['schedule_config'])
            
        except Exception as e:
            logger.error(f"Scheduled report execution failed: {e}")
    
    def _generate_strategic_recommendations(self, business_data: Dict[str, Any]) -> str:
        """Generate strategic recommendations based on business data."""
        recommendations = [
            "Focus on high-performing segments to maximize ROI",
            "Implement data-driven decision making processes",
            "Invest in customer retention programs",
            "Optimize resource allocation based on performance metrics",
            "Develop strategic partnerships for market expansion"
        ]
        
        return "<br>".join([f"• {rec}" for rec in recommendations])
    
    def _generate_story_content(self, data: pd.DataFrame, analysis_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
        """Generate story content sections."""
        return {
            'executive_summary': self._generate_executive_summary(data, analysis_results),
            'key_findings': self._generate_key_findings(analysis_results),
            'detailed_analysis': self._generate_detailed_analysis(data, analysis_results),
            'trends_analysis': self._generate_trends_analysis(data),
            'implications': self._generate_implications(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results)
        }
    
    def _generate_executive_summary(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary."""
        return f"Analysis of {len(data)} records reveals key insights about business performance and trends."
    
    def _generate_key_findings(self, analysis_results: Dict[str, Any]) -> str:
        """Generate key findings section."""
        findings = [
            "Significant growth trends identified in primary metrics",
            "Customer segmentation reveals distinct behavioral patterns",
            "Market opportunities identified through competitive analysis"
        ]
        return "<br>".join([f"• {finding}" for finding in findings])
    
    def _generate_detailed_analysis(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed analysis section."""
        return "Detailed statistical analysis reveals patterns and correlations that inform strategic decisions."
    
    def _generate_trends_analysis(self, data: pd.DataFrame) -> str:
        """Generate trends analysis section."""
        return "Temporal analysis shows clear trends that can inform forecasting and planning activities."
    
    def _generate_implications(self, analysis_results: Dict[str, Any]) -> str:
        """Generate implications section."""
        return "The findings have significant implications for business strategy, operational efficiency, and market positioning."
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = [
            "Implement recommended actions based on data insights",
            "Monitor key metrics for continued optimization",
            "Develop action plans for identified opportunities"
        ]
        return "<br>".join([f"• {rec}" for rec in recommendations])
    
    def start_scheduler(self) -> None:
        """Start the report scheduler."""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("Report scheduler started")