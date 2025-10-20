import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from state.final_state import FinalState
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
import traceback


class ReportLLM:
    """보고서를 생성하는 Agent"""
    
    def __init__(self, final_state: FinalState):
        self.final_state = final_state
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.company_name = final_state["company_name"]
        self._register_fonts()
    
    def _register_fonts(self):
        """한글 폰트 등록"""
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"
            bold_font_path = "C:/Windows/Fonts/malgunbd.ttf"
            
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('Malgun', font_path))
                print(f"✅ 폰트 등록 성공: {font_path}")
            
            if os.path.exists(bold_font_path):
                pdfmetrics.registerFont(TTFont('MalgunBold', bold_font_path))
                print(f"✅ Bold 폰트 등록 성공: {bold_font_path}")
        except Exception as e:
            print(f"❌ 폰트 등록 에러: {e}")
    
    def generate_report(self, output_format: str = "pdf") -> str:
        """
        보고서 생성
        
        Args:
            output_format: 'pdf' 또는 'markdown' (기본값: 'pdf')
        
        Returns:
            PDF 파일 경로 또는 마크다운 텍스트
        """
        
        print("\n보고서 생성 중...")
        
        try:
            # 각 섹션 생성
            overview_section = self._generate_overview_section()
            market_section = self._generate_market_section()
            comparison_section = self._generate_comparison_section()
            tech_section = self._generate_tech_section()
            summary_section = self._generate_summary_section()
            
            # 전체 보고서 구성 (마크다운)
            markdown_report = f"""# 기업 분석 보고서: {self.company_name}

---

## 1. 투자 개요

{overview_section}

---

## 2. 시장성 분석

{market_section}

---

## 3. 경쟁업체 비교

{comparison_section}

---

## 4. 기술력 비교

{tech_section}

---

## 5. 종합 요약

{summary_section}

---

*보고서 생성 일시: {self._get_timestamp()}*
"""
            
            if output_format == "markdown":
                return markdown_report
            elif output_format == "pdf":
                return self._convert_to_pdf(markdown_report)
            else:
                raise ValueError(f"지원하지 않는 형식입니다: {output_format}")
                
        except Exception as e:
            print(f"❌ 보고서 생성 중 에러 발생: {e}")
            traceback.print_exc()
            raise
    
    def _convert_to_pdf(self, markdown_content: str) -> str:
        """마크다운을 PDF로 변환"""
        
        try:
            # PDF 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"report_{self.company_name}_{timestamp}.pdf"
            pdf_path = os.path.join("reports", pdf_filename)
            
            # reports 디렉토리 생성
            os.makedirs("reports", exist_ok=True)
            
            # PDF 문서 생성
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # 스타일 정의
            styles = self._create_styles()
            
            # 컨텐츠 요소 리스트
            story = []
            
            # 마크다운 파싱 및 PDF 요소로 변환
            lines = markdown_content.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                try:
                    # H1 제목
                    if line.startswith('# '):
                        text = line[2:].strip()
                        story.append(Paragraph(text, styles['Title']))
                        story.append(Spacer(1, 0.3 * inch))
                    
                    # H2 제목
                    elif line.startswith('## '):
                        text = line[3:].strip()
                        story.append(Spacer(1, 0.2 * inch))
                        story.append(Paragraph(text, styles['Heading1']))
                        story.append(Spacer(1, 0.15 * inch))
                    
                    # H3 제목
                    elif line.startswith('### '):
                        text = line[4:].strip()
                        story.append(Spacer(1, 0.15 * inch))
                        story.append(Paragraph(text, styles['Heading2']))
                        story.append(Spacer(1, 0.1 * inch))
                    
                    # H4 제목
                    elif line.startswith('#### '):
                        text = line[5:].strip()
                        story.append(Paragraph(text, styles['Heading3']))
                        story.append(Spacer(1, 0.08 * inch))
                    
                    # 구분선
                    elif line.startswith('---'):
                        story.append(Spacer(1, 0.2 * inch))
                    
                    # 리스트 항목
                    elif line.startswith('- ') or line.startswith('* '):
                        text = line[2:].strip()
                        # 강조 처리
                        text = self._process_markdown_emphasis(text)
                        story.append(Paragraph(f"• {text}", styles['Bullet']))
                        story.append(Spacer(1, 0.05 * inch))
                    
                    # 일반 문단
                    elif line and not line.startswith('#'):
                        # 강조 처리
                        text = self._process_markdown_emphasis(line)
                        story.append(Paragraph(text, styles['Body']))
                        story.append(Spacer(1, 0.1 * inch))
                
                except Exception as e:
                    print(f"⚠️ 라인 처리 중 에러 (라인 {i}): {e}")
                    # 에러가 발생해도 계속 진행
                
                i += 1
            
            # PDF 생성
            doc.build(story)
            
            return pdf_path
            
        except Exception as e:
            print(f"❌ PDF 변환 중 에러: {e}")
            traceback.print_exc()
            raise
    
    def _process_markdown_emphasis(self, text: str) -> str:
        """마크다운 강조 표시를 HTML로 변환"""
        # **굵게** -> <b>굵게</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # *기울임* -> <i>기울임</i>
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        # __굵게__ -> <b>굵게</b>
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        return text
    
    def _create_styles(self):
        """PDF 스타일 생성"""
        styles = getSampleStyleSheet()
        
        # 한글 폰트 사용
        korean_font = 'Malgun' if 'Malgun' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
        korean_bold = 'MalgunBold' if 'MalgunBold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
        
        # 기존 스타일 수정 (add 대신 직접 속성 변경)
        styles['Title'].fontName = korean_bold
        styles['Title'].fontSize = 24
        styles['Title'].textColor = colors.HexColor('#1a1a1a')
        styles['Title'].spaceAfter = 20
        styles['Title'].alignment = TA_LEFT
        
        styles['Heading1'].fontName = korean_bold
        styles['Heading1'].fontSize = 18
        styles['Heading1'].textColor = colors.HexColor('#2c3e50')
        styles['Heading1'].spaceAfter = 12
        styles['Heading1'].spaceBefore = 20
        
        styles['Heading2'].fontName = korean_bold
        styles['Heading2'].fontSize = 14
        styles['Heading2'].textColor = colors.HexColor('#34495e')
        styles['Heading2'].spaceAfter = 10
        styles['Heading2'].spaceBefore = 15
        
        # Heading3 처리
        if 'Heading3' in styles:
            styles['Heading3'].fontName = korean_bold
            styles['Heading3'].fontSize = 12
            styles['Heading3'].textColor = colors.HexColor('#555555')
            styles['Heading3'].spaceAfter = 8
            styles['Heading3'].spaceBefore = 12
        else:
            styles.add(ParagraphStyle(
                name='Heading3',
                fontName=korean_bold,
                fontSize=12,
                textColor=colors.HexColor('#555555'),
                spaceAfter=8,
                spaceBefore=12
            ))
        
        # Body 스타일
        if 'Body' not in styles:
            styles.add(ParagraphStyle(
                name='Body',
                parent=styles['Normal'],
                fontName=korean_font,
                fontSize=10,
                leading=16,
                textColor=colors.HexColor('#333333'),
                alignment=TA_JUSTIFY,
                wordWrap='CJK'
            ))
        
        # Bullet 스타일
        if 'Bullet' not in styles:
            styles.add(ParagraphStyle(
                name='Bullet',
                parent=styles['Normal'],
                fontName=korean_font,
                fontSize=10,
                leading=14,
                leftIndent=20,
                textColor=colors.HexColor('#333333'),
                wordWrap='CJK'
            ))
        
        return styles
    
    def _generate_overview_section(self) -> str:
        """1. 투자 개요 섹션 생성"""
        
        market_state = self.final_state.get("market")
        tech_state = self.final_state.get("tech")
        comparison_state = self.final_state.get("comparison")
        
        # 점수 계산
        scores = []
        if market_state:
            scores.append(market_state.get("competitor_score", 0))
        if tech_state:
            scores.append(tech_state.get("technology_score", 0))
        if comparison_state:
            scores.append(comparison_state.get("competitor_score", 0))
        
        average_score = sum(scores) / len(scores) if scores else 0
        
        prompt = f"""당신은 벤처 투자 전문가입니다. 다음 정보를 바탕으로 투자 개요 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}
평균 평가 점수: {average_score:.1f}점

=== 개별 점수 ===
시장성 점수: {market_state.get('competitor_score', 'N/A') if market_state else 'N/A'}점
기술력 점수: {tech_state.get('technology_score', 'N/A') if tech_state else 'N/A'}점
경쟁력 점수: {comparison_state.get('competitor_score', 'N/A') if comparison_state else 'N/A'}점

다음 형식으로 작성해주세요:

### 1.1 투자요약
회사의 전반적인 투자 매력도를 3-4문단으로 요약합니다. 평균 점수와 각 영역별 점수를 근거로 작성하고, 투자자 관점에서의 핵심 가치 제안을 포함합니다.

### 1.2 핵심 인사이트 및 레드플래그

**주요 강점:**
1. [구체적인 강점 1]
2. [구체적인 강점 2]
3. [구체적인 강점 3]

**주의해야 할 리스크 또는 개선 필요 사항:**
1. [리스크 또는 개선사항 1]
2. [리스크 또는 개선사항 2]
3. [리스크 또는 개선사항 3]

**중요**: 
- 반드시 마크다운 형식으로 작성
- 전문적이고 객관적인 톤 유지
- 각 항목은 구체적인 근거와 함께 작성
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ LLM 호출 에러 (overview): {e}")
            return "### 1.1 투자요약\n데이터를 생성할 수 없습니다.\n\n### 1.2 핵심 인사이트\n데이터 부족"
    
    def _generate_market_section(self) -> str:
        """2. 시장성 분석 섹션 생성"""
        
        market_state = self.final_state.get("market")
        
        if not market_state:
            return """### 2.1 문제정의·고객세그먼트
시장 분석 데이터가 없습니다.

### 2.2 규제·거시 트렌드
시장 분석 데이터가 없습니다."""
        
        market_score = market_state.get("competitor_score", 0)
        market_basis = market_state.get("competitor_analysis_basis", "")
        
        prompt = f"""당신은 시장 분석 전문가입니다. 다음 정보를 바탕으로 시장성 분석 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}

=== 시장 평가 결과 ===
시장 점수: {market_score}점
평가 근거:
{market_basis}

다음 형식으로 작성해주세요:

### 2.1 문제정의·고객세그먼트
회사가 해결하려는 문제와 타겟 고객층을 분석합니다. 시장 기회의 크기와 성장 가능성, 고객 니즈와 솔루션의 적합성을 3-4문단으로 구성합니다.

### 2.2 규제·거시 트렌드
관련 산업의 규제 환경, 거시 경제 및 산업 트렌드, 정책적 지원 및 리스크 요인을 3-4문단으로 구성합니다.

**중요**: 
- 마크다운 형식으로 작성
- 평가 근거에 제시된 구체적인 정보를 활용
- 객관적이고 전문적인 분석
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ LLM 호출 에러 (market): {e}")
            return "### 2.1 문제정의\n데이터를 생성할 수 없습니다.\n\n### 2.2 규제 트렌드\n데이터를 생성할 수 없습니다."
    
    def _generate_comparison_section(self) -> str:
        """3. 경쟁업체 비교 섹션 생성"""
        
        comparison_state = self.final_state.get("comparison")
        
        if not comparison_state:
            return """### 3.1 경쟁지표(벤치마크, 대체재)
경쟁사 비교 데이터가 없습니다.

### 3.2 포지셔닝 맵·진입장벽
경쟁사 비교 데이터가 없습니다."""
        
        comparison_score = comparison_state.get("competitor_score", 0)
        comparison_basis = comparison_state.get("competitor_analysis_basis", "")
        
        prompt = f"""당신은 경쟁 분석 전문가입니다. 다음 정보를 바탕으로 경쟁업체 비교 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}

=== 경쟁사 비교 결과 ===
경쟁력 점수: {comparison_score}점
평가 근거:
{comparison_basis}

다음 형식으로 작성해주세요:

### 3.1 경쟁지표(벤치마크, 대체재)
주요 경쟁사 대비 우위 및 열위 항목, 재무지표와 사용자 지표(MUV 등) 비교 분석을 평가 근거에 포함된 표나 데이터를 활용하여 구체적으로 3-4문단으로 작성합니다.

### 3.2 포지셔닝 맵·진입장벽
시장 내 포지셔닝 및 차별화 요소, 경쟁사 대비 진입장벽 및 방어 가능성, 향후 경쟁 구도 전망을 3-4문단으로 구성합니다.

**중요**: 
- 마크다운 형식으로 작성
- 평가 근거의 구체적인 수치와 비교 데이터 활용
- 객관적이고 분석적인 톤 유지
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ LLM 호출 에러 (comparison): {e}")
            return "### 3.1 경쟁지표\n데이터를 생성할 수 없습니다.\n\n### 3.2 포지셔닝\n데이터를 생성할 수 없습니다."
    
    def _generate_tech_section(self) -> str:
        """4. 기술력 비교 섹션 생성"""
        
        tech_state = self.final_state.get("tech")
        
        if not tech_state:
            return """### 4.1 제품 로드맵·사용자 여정
기술 분석 데이터가 없습니다.

### 4.2 품질/안전
기술 분석 데이터가 없습니다."""
        
        tech_score = tech_state.get("technology_score", 0)
        tech_basis = tech_state.get("technology_analysis_basis", "")
        category_scores = tech_state.get("category_scores", {})
        
        prompt = f"""당신은 기술 분석 전문가입니다. 다음 정보를 바탕으로 기술력 비교 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}

=== 기술 평가 결과 ===
기술 점수: {tech_score}점
평가 근거:
{tech_basis}

항목별 점수:
- 혁신성(Innovation): {category_scores.get('innovation', 0)}/30점
- 완성도(Completeness): {category_scores.get('completeness', 0)}/30점
- 경쟁력(Competitiveness): {category_scores.get('competitiveness', 0)}/20점
- 특허(Patent): {category_scores.get('patent', 0)}/10점
- 확장성(Scalability): {category_scores.get('scalability', 0)}/10점

다음 형식으로 작성해주세요:

### 4.1 제품 로드맵·사용자 여정
회사의 기술 발전 방향과 제품 로드맵 분석, 사용자 여정 관점에서의 기술 적용 사례를 평가 근거에 언급된 구체적인 기술과 서비스를 활용하여 3-4문단으로 구성합니다.

### 4.2 품질/안전
기술 점수 및 항목별 평가 요약, 품질 관리 체계 및 안전성 평가, 기술적 강점과 개선 필요 사항, 특허 및 기술 자산 분석을 3-4문단으로 구성합니다.

**중요**: 
- 마크다운 형식으로 작성
- 항목별 점수를 근거로 구체적으로 분석
- 전문적이고 기술적인 톤 유지
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ LLM 호출 에러 (tech): {e}")
            return "### 4.1 제품 로드맵\n데이터를 생성할 수 없습니다.\n\n### 4.2 품질/안전\n데이터를 생성할 수 없습니다."
    
    def _generate_summary_section(self) -> str:
        """5. 종합 요약 섹션 생성"""
        
        market_state = self.final_state.get("market")
        tech_state = self.final_state.get("tech")
        comparison_state = self.final_state.get("comparison")
        
        # 점수 계산
        scores = []
        score_details = []
        
        if market_state:
            market_score = market_state.get("competitor_score", 0)
            scores.append(market_score)
            score_details.append(f"시장성: {market_score}점")
        
        if tech_state:
            tech_score = tech_state.get("technology_score", 0)
            scores.append(tech_score)
            score_details.append(f"기술력: {tech_score}점")
        
        if comparison_state:
            comparison_score = comparison_state.get("competitor_score", 0)
            scores.append(comparison_score)
            score_details.append(f"경쟁력: {comparison_score}점")
        
        average_score = sum(scores) / len(scores) if scores else 0
        
        prompt = f"""당신은 벤처 투자 의사결정 전문가입니다. 다음 정보를 바탕으로 종합 요약 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}

=== 종합 평가 결과 ===
평균 점수: {average_score:.1f}점
개별 점수: {', '.join(score_details)}

=== 각 영역 평가 근거 ===

[시장성]
{market_state.get('competitor_analysis_basis', 'N/A') if market_state else 'N/A'}

[기술력]
{tech_state.get('technology_analysis_basis', 'N/A') if tech_state else 'N/A'}

[경쟁력]
{comparison_state.get('competitor_analysis_basis', 'N/A') if comparison_state else 'N/A'}

다음 형식으로 작성해주세요:

### 5.1 기업 투자 순위
평균 점수를 기반으로 투자 등급을 제시합니다(S/A/B/C/D 등급). 등급 산정 근거를 각 영역별 점수와 함께 설명하고, 동종 업계 내 상대적 위치를 평가합니다. 2-3문단으로 구성합니다.

### 5.2 기업 종합 평가 및 투자 전략
투자 매력도 종합 평가, 강점과 약점 요약, 구체적인 투자 전략 제안(투자 시기, 규모, 조건 등), 리스크 관리 방안, Exit 전략 고려사항을 4-5문단으로 구성합니다.

**중요**: 
- 마크다운 형식으로 작성
- 모든 영역의 평가를 종합하여 균형잡힌 결론 도출
- 투자자 관점에서 실행 가능한 전략 제시
- 전문적이고 설득력 있는 톤 유지
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ LLM 호출 에러 (summary): {e}")
            return "### 5.1 기업 투자 순위\n데이터를 생성할 수 없습니다.\n\n### 5.2 종합 평가\n데이터를 생성할 수 없습니다."
    
    def _get_timestamp(self):
        """현재 시각 반환"""
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")