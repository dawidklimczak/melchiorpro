import streamlit as st
from openai import OpenAI
import json
import os
import re
import time
from typing import List, Dict, Any, Optional
import base64

# Stałe i schematy
SETTINGS_FILE = "settings.json"

def default_settings() -> Dict[str, Any]:
    """Zwraca domyślne ustawienia."""
    return {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 2000,
        "do_research": False,
        "search_context_size": "medium",
        "add_bibliography": True,
        "search_intent": "informacyjna",
        "complexity_level": "średniozaawansowany",
        "engagement_elements": {
            "rhetorical_questions": True,
            "statistics_quotes": True,
            "examples_cases": True,
            "stories": False
        },
        "content_type": "przewodnik",
        "readability_index": "standardowy",
        "num_topics": 5,
        "num_sections": 7,
        "section_length": "średnia (250-350 słów)"
    }

# Schema dla tematów
TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "description": "Lista proponowanych tematów artykułów",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Unikalny identyfikator tematu",
                        "minimum": 1
                    },
                    "title": {
                        "type": "string",
                        "description": "Tytuł artykułu zoptymalizowany pod SEO"
                    },
                    "keywords": {
                        "type": "array",
                        "description": "Lista słów kluczowych",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "description": {
                        "type": "string",
                        "description": "Krótki opis tematu artykułu"
                    }
                },
                "required": ["id", "title", "keywords", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["topics"],
    "additionalProperties": False
}

# Funkcje pomocnicze
def save_settings(settings: Dict[str, Any]) -> None:
    """Zapisuje ustawienia do pliku JSON."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def load_settings() -> Dict[str, Any]:
    """Wczytuje ustawienia z pliku JSON."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return default_settings()
    except Exception as e:
        st.warning(f"Nie udało się wczytać ustawień: {str(e)}")
        return default_settings()

class ArticleGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.research_results = {}

    def generate_topics(self, keywords: str, settings: Dict[str, Any]) -> List[Dict]:
        """Generuje propozycje tematów na podstawie słów kluczowych."""
        search_intent = settings.get("search_intent", "informacyjna")
        complexity_level = settings.get("complexity_level", "średniozaawansowany")
        content_type = settings.get("content_type", "przewodnik")
        num_topics = settings.get("num_topics", 5)
        
        system_prompt = f"""
        Jesteś ekspertem SEO i content marketingu. Generuj propozycje tematów artykułów.
        
        Intencja wyszukiwania: {search_intent}
        Poziom złożoności: {complexity_level}
        Typ treści: {content_type}
        
        Odpowiedź musi być w następującym formacie JSON:
        {{
            "topics": [
                {{
                    "id": 1,
                    "title": "Tytuł artykułu (max 60 znaków)",
                    "keywords": ["słowo1", "słowo2", "słowo3"],
                    "description": "Krótki opis tematu (max 200 znaków)"
                }}
            ]
        }}
        
        Wymagania:
        - Dokładnie {num_topics} tematów
        - Każdy temat musi mieć unikalny id (1-{num_topics})
        - Tytuł musi być chwytliwy i SEO-friendly
        - 3-5 słów kluczowych na temat
        - Krótki ale treściwy opis
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Wygeneruj {num_topics} tematów artykułów na podstawie słów kluczowych: {keywords}"
            }],
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 1.0),
            max_tokens=settings.get("max_tokens", 2000),
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)["topics"]
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania odpowiedzi: {str(e)}")
            return []

    def perform_web_research(self, topic: Dict, settings: Dict[str, Any]) -> str:
        """Wykonuje research w internecie na temat danego tematu."""
        if not settings.get("do_research", False):
            return ""
            
        search_context_size = settings.get("search_context_size", "medium")
        st.info("Rozpoczynanie researchu internetowego...")
        
        try:
            st.info("Używam modelu gpt-4o-mini-search-preview z web_search_options...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                web_search_options={
                    "search_context_size": search_context_size
                },
                messages=[
                    {
                        "role": "system",
                        "content": """Jesteś asystentem badawczym specjalizującym się w dostarczaniu wyczerpujących i dobrze udokumentowanych informacji.
                        
                        WAŻNE:
                        1. Zbierz informacje z MINIMUM 4-5 RÓŻNYCH ŹRÓDEŁ.
                        2. Korzystaj z różnorodnych źródeł: czasopisma naukowe, szanowane magazyny, raporty branżowe, statystyki rządowe.
                        3. Cytuj każde źródło, z którego korzystasz.
                        4. Prezentuj zróżnicowane perspektywy na temat.
                        5. Podaj najnowsze dostępne dane, badania i przykłady.
                        
                        Twój research musi zawierać:
                        - Aktualne statystyki i dane
                        - Opinie i komentarze ekspertów (z podaniem ich afiliacji)
                        - Przykłady i studia przypadków
                        - Najnowsze trendy w tej dziedzinie
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Przeprowadź szczegółowy research na temat: '{topic['title']}'. Słowa kluczowe do uwzględnienia: {', '.join(topic['keywords'])}. Zbierz różnorodne informacje z WIELU różnych źródeł i upewnij się, że każde źródło jest wyraźnie oznaczone."
                    }
                ]
            )
            
            content = response.choices[0].message.content
            
            # Sprawdzamy czy mamy annotations z cytowaniami
            annotations = []
            if hasattr(response.choices[0].message, 'annotations'):
                annotations = response.choices[0].message.annotations
            
            citations = []
            for annotation in annotations:
                if hasattr(annotation, 'url_citation'):
                    citations.append(annotation.url_citation.url)
            
            # Jeśli nie ma citations z annotations, próbujemy wyciągnąć z tekstu
            if not citations:
                url_pattern = r'https?://[^\s)"\']+'
                citations = list(set(re.findall(url_pattern, content)))
            
            # Wyodrębnij nazwy domen dla podsumowania
            domains = []
            for url in citations:
                try:
                    domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
                    domains.append(domain)
                except:
                    pass
            
            num_sources = len(set(domains))
            st.success(f"Research zakończony pomyślnie! Znaleziono dane z {num_sources} różnych źródeł.")
            
            # Przechowujemy wyniki badania
            self.research_results[topic["id"]] = {
                "content": content,
                "citations": citations
            }
            
            return content
                
        except Exception as e:
            st.error(f"Nie udało się przeprowadzić researchu online: {str(e)}")
            st.warning("Research internetowy niedostępny. Sprawdź wersję biblioteki OpenAI (min. 1.10.0) i upewnij się, że masz dostęp do modeli z funkcją wyszukiwania.")
            
            if st.button("Pokaż szczegóły błędu"):
                st.code(str(e), language="python")
                
                st.markdown("""
                ### Wymagania dla korzystania z web_search_options:
                1. Wersja biblioteki OpenAI min. 1.10.0
                2. Dostęp do modelu gpt-4o-mini-search-preview (może wymagać specjalnego dostępu)
                3. Prawidłowy klucz API z dostępem do Search API
                """)
            
            return ""

    def generate_article_outline(self, topic: Dict, settings: Dict[str, Any]) -> Dict:
        """Generuje konspekt artykułu."""
        search_intent = settings.get("search_intent", "informacyjna")
        complexity_level = settings.get("complexity_level", "średniozaawansowany")
        content_type = settings.get("content_type", "przewodnik")
        engagement_elements = settings.get("engagement_elements", {})
        num_sections = settings.get("num_sections", 7)
        section_length = settings.get("section_length", "średnia (250-350 słów)")
        
        # Określenie typowej długości sekcji na podstawie ustawienia
        words_range = ""
        if "krótka" in section_length:
            words_range = "200-250"
            target_words = 225
        elif "długa" in section_length:
            words_range = "350-450"
            target_words = 400
        else:
            words_range = "250-350"
            target_words = 300
        
        engagement_instructions = []
        if engagement_elements.get("rhetorical_questions", False):
            engagement_instructions.append("Wykorzystaj pytania retoryczne do angażowania czytelnika")
        if engagement_elements.get("statistics_quotes", False):
            engagement_instructions.append("Włącz dane statystyczne i cytaty ekspertów")
        if engagement_elements.get("examples_cases", False):
            engagement_instructions.append("Dodaj przykłady i case studies")
        if engagement_elements.get("stories", False):
            engagement_instructions.append("Użyj historii i anegdot dla ilustracji punktów")
            
        engagement_text = "\n- " + "\n- ".join(engagement_instructions) if engagement_instructions else ""
        
        # Dodajemy dane z researchu, jeśli dostępne
        research_data = self.research_results.get(topic["id"], {}).get("content", "")
        reference_content = ""
        if research_data:
            reference_content = "Dane z researchu:\n" + research_data
        
        system_prompt = f"""
        Jesteś ekspertem w tworzeniu konspektów artykułów. 
        
        Intencja wyszukiwania: {search_intent}
        Poziom złożoności: {complexity_level}
        Typ treści: {content_type}
        Elementy angażujące: {engagement_text}
        
        Odpowiedź musi być w następującym formacie JSON:
        {{
            "article_title": "Finalny tytuł artykułu",
            "target_audience": "Szczegółowy opis grupy docelowej",
            "main_keywords": ["słowo1", "słowo2", "słowo3", "słowo4", "słowo5"],
            "sections": [
                {{
                    "section_number": 1,
                    "title": "Tytuł sekcji",
                    "estimated_words": {target_words},
                    "keywords": ["słowo1", "słowo2"],
                    "prompt": "Szczegółowe instrukcje do wygenerowania sekcji"
                }}
            ]
        }}
        
        WAŻNE INSTRUKCJE:
        1. Wygeneruj konspekt dla SPÓJNEGO artykułu, który będzie czytany jako jednolity tekst.
        2. Utwórz DOKŁADNIE {num_sections} sekcji (numerowanych od 1 do {num_sections}).
        3. Pierwsza sekcja to wstęp, który zaciekawi czytelnika.
        4. Ostatnia sekcja to podsumowanie, które zbierze najważniejsze punkty.
        5. Każda sekcja powinna mieć unikalny tytuł.
        6. Tytuły sekcji muszą być zapisane naturalnie - nie każde słowo z wielkiej litery.
        7. Każda sekcja powinna zawierać 2-3 słowa kluczowe.
        8. Każda sekcja powinna mieć długość dokładnie {target_words} słów.
        
        INSTRUKCJE DOTYCZĄCE PROMPTÓW:
        - Prompty dla sekcji muszą być BARDZO SZCZEGÓŁOWE i obszerne (minimum 200 znaków).
        - Każdy prompt powinien zawierać:
          a) Dokładny cel sekcji
          b) 3-5 kluczowych punktów do omówienia
          c) Sugerowany ton i styl wypowiedzi
          d) Konkretne informacje, jakie powinny się znaleźć w sekcji
          e) Wskazówki dotyczące struktury sekcji (np. rozpocznij od przykładu, zakończ pytaniem)
        - Prompty nie mogą być ogólnikowe - powinny dostarczać konkretnych wskazówek.
        - Zaznacz w promptach, że każda sekcja powinna używać akapitów (paragrafów) dla lepszej czytelności.
        
        DODATKOWE WYMAGANIA:
        1. Sekcje powinny LOGICZNIE wynikać z siebie, tworząc płynną narrację.
        2. NIE twórz podtytułów ani podsekcji - każda sekcja to jeden ciągły blok tekstu z akapitami.
        3. Tytuł artykułu musi być chwytliwy, SEO-friendly i jasno komunikować temat.
        4. Upewnij się, że w promptach zaznaczasz, aby unikać powtarzania struktur gramatycznych na początku każdej sekcji (np. aby nie zaczynać każdej sekcji od "Film XYZ to..." lub podobnego szablonu).
        
        WAŻNE: Wykorzystaj dane z researchu, jeśli są dostępne, aby stworzyć kompletny i dobrze zaplanowany konspekt.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Stwórz konspekt artykułu.\nTemat: {topic['title']}\n"
                          f"Słowa kluczowe: {', '.join(topic['keywords'])}\n"
                          f"Opis: {topic['description']}\n"
                          f"Liczba sekcji: DOKŁADNIE {num_sections}\n"
                          f"Treść referencyjna: {reference_content}"
            }],
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 1.0),
            max_tokens=settings.get("max_tokens", 2000),
            response_format={"type": "json_object"}
        )
        
        try:
            outline = json.loads(response.choices[0].message.content)
            
            # Sprawdzenie, czy liczba sekcji się zgadza
            if len(outline.get("sections", [])) != num_sections:
                st.warning(f"Model wygenerował {len(outline.get('sections', []))} sekcji zamiast żądanych {num_sections}. Spróbuj ponownie lub dostosuj liczbę sekcji.")
            
            # Sprawdzenie długości promptów
            short_prompts = [section for section in outline.get("sections", []) if len(section.get("prompt", "")) < 200]
            if short_prompts:
                st.warning(f"Niektóre prompty są zbyt krótkie ({len(short_prompts)} z {num_sections}). Jakość artykułu może być niższa.")
            
            return outline
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania konspektu: {str(e)}")
            return None

    def generate_article_section_by_section(self, outline: Dict, topic: Dict, settings: Dict[str, Any]) -> str:
        """Generuje artykuł sekcja po sekcji z zawyżonymi wartościami długości."""
        # Pobierz dane z researchu, jeśli istnieją
        research_content = self.research_results.get(topic["id"], {}).get("content", "")
        
        # Parametry artykułu
        section_length = settings.get("section_length", "średnia (250-350 słów)")
        
        # Współczynnik zwiększający dla długości sekcji
        length_multiplier = 1.25  # Prosimy o 25% więcej słów niż faktycznie potrzebujemy
        
        # Określenie rzeczywistej i docelowej długości sekcji
        if "krótka" in section_length:
            real_target = 225
            display_target = real_target  # Co pokazujemy użytkownikowi
            ai_target = int(real_target * length_multiplier)  # O co prosimy AI (około 280 słów)
            min_acceptable = 180  # Minimum akceptowalne (80% z rzeczywistego celu)
        elif "długa" in section_length:
            real_target = 400
            display_target = real_target
            ai_target = int(real_target * length_multiplier)  # O co prosimy AI (około 500 słów)
            min_acceptable = 320  # Minimum akceptowalne (80% z rzeczywistego celu)
        else:  # średnia
            real_target = 300
            display_target = real_target
            ai_target = int(real_target * length_multiplier)  # O co prosimy AI (około 375 słów)
            min_acceptable = 240  # Minimum akceptowalne (80% z rzeczywistego celu)
        
        # Inicjalizacja pełnego artykułu
        full_article = f"<h3>{outline['article_title']}</h3>\n\n"
        total_words = 0
        
        # Progressbar dla całego procesu
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Generowanie sekcji pojedynczo
        for i, section in enumerate(outline['sections']):
            # Aktualizacja postępu
            progress = i / len(outline['sections'])
            progress_bar.progress(progress)
            progress_text.text(f"Generowanie sekcji {i+1} z {len(outline['sections'])}: {section['title']}")
            
            # Stwórz kontekst wcześniejszych sekcji (skrócona wersja dla zachowania kontekstu)
            previous_context = full_article[-4000:] if len(full_article) > 4000 else full_article
            
            # Prompt dla pojedynczej sekcji
            system_prompt = f"""
            Jesteś profesjonalnym copywriterem. Twoim zadaniem jest napisanie JEDNEJ SEKCJI artykułu.
            
            WYMAGANIA DŁUGOŚCI - BARDZO WAŻNE:
            1. Ta sekcja MUSI mieć DOKŁADNIE {ai_target} słów - TO JEST ABSOLUTNY WYMÓG.
            2. Minimum to {ai_target - 50} słów.
            3. Po napisaniu POLICZ dokładnie słowa i jeśli jest mniej niż {ai_target}, WYDŁUŻ tekst.
            
            SPECYFIKACJA SEKCJI:
            - Tytuł sekcji: {section['title']}
            - Słowa kluczowe: {', '.join(section['keywords'])}
            - Cel: {section['prompt']}
            - Numer sekcji: {section['section_number']} z {len(outline['sections'])}
            
            FORMAT:
            <h3>{section['title']}</h3>
            [Tutaj dokładnie {ai_target} słów treści sekcji bez dodatkowych nagłówków]
            
            STYL I TREŚĆ:
            1. Tekst musi być szczegółowy, informatywny i wartościowy.
            2. Używaj naturalnych wtrąceń słów kluczowych.
            3. Unikaj powtórzeń i wypełniaczy.
            4. Nie twórz podsekcji ani podtytułów - tylko jeden ciągły tekst.
            5. Dostosuj się do wcześniejszych sekcji dla zachowania spójności.
            6. Podziel tekst na paragrafy używając znaczników <p> i </p> dla lepszej czytelności.
            7. UNIKAJ powtórzenia tych samych struktur gramatycznych na początku sekcji. Nie zaczynaj każdej sekcji według tego samego szablonu (np. "Film X to...", "Technologia Y jest...").
            8. Wprowadzaj różnorodność w stylu rozpoczynania sekcji - używaj pytań, cytatów, anegdot, danych statystycznych, itp.
            9. NIE DODAWAJ na końcu sekcji informacji o liczbie słów (np. "500 słów").
            
            BARDZO WAŻNE:
            - Generuj TYLKO tę jedną sekcję, nie cały artykuł.
            - Nie powtarzaj treści z poprzednich sekcji.
            - UPEWNIJ SIĘ, że sekcja ma co najmniej {ai_target - 50} słów.
            - Przed zakończeniem POLICZ SŁOWA i sprawdź, czy jest ich wystarczająco dużo.
            """
            
            # Generowanie sekcji - tylko jedna próba
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"""Napisz sekcję "{section['title']}" dla artykułu "{outline['article_title']}".
                        
                        Grupa docelowa: {outline['target_audience']}
                        
                        Wcześniejsze sekcje artykułu (dla kontekstu):
                        {previous_context}
                        
                        Dane z researchu (wykorzystaj te informacje):
                        {research_content[:2000] if research_content else "Brak danych z researchu."}
                        
                        WAŻNE: Ta sekcja MUSI zawierać DOKŁADNIE {ai_target} słów. Policz dokładnie słowa przed zakończeniem.
                        """
                    }
                ],
                temperature=settings.get("temperature", 0.8),
                top_p=settings.get("top_p", 1.0),
                max_tokens=2000
            )
            
            section_content = response.choices[0].message.content
            
            # Usuwamy tagi HTML do liczenia słów
            clean_text = re.sub(r'<[^>]+>', '', section_content)
            section_word_count = len(clean_text.split())
            
            # Wyświetl informację o długości (pokazujemy użytkownikowi wartości, których oczekiwał)
            if section_word_count >= min_acceptable:
                st.success(f"Sekcja {i+1}: {section_word_count} słów (cel: {display_target})")
            else:
                st.warning(f"Sekcja {i+1}: {section_word_count} słów (cel: {display_target}) - trochę krótsza niż oczekiwano")
            
            # Dodaj sekcję do artykułu niezależnie od długości - jedna próba
            # Usuwamy nagłówek h3 jeśli został dodany przez model, aby uniknąć duplikatów
            section_content = re.sub(r'<h3>.*?</h3>\s*', '', section_content)
            
            # Dodajemy własny nagłówek i sekcję
            full_article += f"<h3>{section['title']}</h3>\n\n{section_content}\n\n"
            total_words += section_word_count
        
        # Zakończenie procesu
        progress_bar.progress(1.0)
        progress_text.text(f"Zakończono generowanie artykułu: {total_words} słów")
        
        # Dodaj bibliografię, jeśli potrzebna
        citations = []
        if topic["id"] in self.research_results:
            citations = self.research_results[topic["id"]].get("citations", [])
            
        if settings.get("add_bibliography", True) and citations:
            bibliography = "\n<h3>Bibliografia</h3>\n<ol>\n"
            for citation in citations:
                # Czyszczenie URL z parametrów
                if citation.startswith('http'):
                    # Usuwamy parametry query
                    clean_url = citation.split("?")[0] if "?" in citation else citation
                    bibliography += f'    <li><a href="{clean_url}" target="_blank">{clean_url}</a></li>\n'
                else:
                    # To jest cytowanie tekstowe
                    bibliography += f"    <li>{citation}</li>\n"
            bibliography += "</ol>\n"
            full_article += bibliography
        
        # Informacja o całkowitej długości
        st.success(f"Wygenerowano artykuł o łącznej długości {total_words} słów.")
        
        return full_article

# Funkcja do tworzenia przycisku kopiowania do schowka
def get_copy_to_clipboard_button(text):
    b64 = base64.b64encode(text.encode()).decode()
    return f"""
    <button onclick="navigator.clipboard.writeText(atob('{b64}'))">
        Kopiuj źródło do schowka
    </button>
    """

# Funkcja główna aplikacji
def main():
    st.set_page_config(page_title="Melchior", layout="wide")

    # Inicjalizacja stanu sesji
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'generator' not in st.session_state:
        st.session_state.generator = ArticleGenerator()
    if 'selected_topic_index' not in st.session_state:
        st.session_state.selected_topic_index = None
    if 'outline' not in st.session_state:
        st.session_state.outline = None
    if 'article_content' not in st.session_state:
        st.session_state.article_content = None
    if 'html_view' not in st.session_state:
        st.session_state.html_view = "podgląd"  # podgląd lub źródło
        
    # Nagłówek
    st.title("Melchior - Generator Artykułów")
    
    # Tworzenie struktury dwukolumnowej
    col1, col2 = st.columns([1, 1])
    
    # Panel ustawień (lewa kolumna)
    with col1:
        st.header("Panel ustawień")
        
        # Parametry merytoryczne
        with st.expander("Parametry merytoryczne", expanded=True):
            # Intencja wyszukiwania
            search_intent = st.radio(
                "Intencja wyszukiwania",
                options=["informacyjna", "nawigacyjna", "transakcyjna", "komercyjna"],
                index=["informacyjna", "nawigacyjna", "transakcyjna", "komercyjna"].index(st.session_state.settings.get("search_intent", "informacyjna")),
                horizontal=True
            )
            
            # Poziom złożoności
            complexity_level = st.select_slider(
                "Poziom złożoności",
                options=["podstawowy", "średniozaawansowany", "ekspercki"],
                value=st.session_state.settings.get("complexity_level", "średniozaawansowany")
            )
            
            # Elementy angażujące
            st.subheader("Elementy angażujące")
            engagement_elements = st.session_state.settings.get("engagement_elements", {})
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                rhetorical_questions = st.checkbox(
                    "Pytania retoryczne",
                    value=engagement_elements.get("rhetorical_questions", True)
                )
                statistics_quotes = st.checkbox(
                    "Dane statystyczne i cytaty",
                    value=engagement_elements.get("statistics_quotes", True)
                )
                
            with col_e2:
                examples_cases = st.checkbox(
                    "Przykłady i case studies",
                    value=engagement_elements.get("examples_cases", True)
                )
                stories = st.checkbox(
                    "Historie i anegdoty",
                    value=engagement_elements.get("stories", False)
                )
                
            # Typ treści
            content_type = st.selectbox(
                "Typ treści",
                options=["how-to", "lista", "przewodnik", "case study", "recenzja", "porównanie", "FAQ"],
                index=["how-to", "lista", "przewodnik", "case study", "recenzja", "porównanie", "FAQ"].index(st.session_state.settings.get("content_type", "przewodnik"))
            )
            
            # Wskaźnik czytelności
            readability_index = st.select_slider(
                "Wskaźnik czytelności",
                options=["prostszy", "standardowy", "bardziej złożony"],
                value=st.session_state.settings.get("readability_index", "standardowy")
            )
            
            # Liczba tematów
            num_topics = st.number_input(
                "Liczba tematów",
                min_value=3,
                max_value=10,
                value=st.session_state.settings.get("num_topics", 5)
            )
            
            # Parametry długości artykułu
            st.subheader("Parametry długości artykułu")
            col_l1, col_l2 = st.columns(2)
            
            with col_l1:
                num_sections = st.number_input(
                    "Liczba sekcji",
                    min_value=3,
                    max_value=15,
                    value=st.session_state.settings.get("num_sections", 7)
                )
            
            # Lista dostępnych opcji długości sekcji
            section_length_options = ["krótka (200-250 słów)", "średnia (250-350 słów)", "długa (350-450 słów)"]
            
            # Sprawdzenie, czy zapisana wartość jest na liście dostępnych opcji
            saved_section_length = st.session_state.settings.get("section_length", "średnia (250-350 słów)")
            if saved_section_length not in section_length_options:
                # Jeśli nie, używamy wartości domyślnej
                saved_section_length = "średnia (250-350 słów)"
            
            with col_l2:
                section_length = st.select_slider(
                    "Długość sekcji",
                    options=section_length_options,
                    value=saved_section_length
                )
            
        # Parametry techniczne
        with st.expander("Parametry techniczne", expanded=True):
            # Temperatura
            temperature = st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings.get("temperature", 0.7),
                step=0.1
            )
            
            # Top P
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings.get("top_p", 1.0),
                step=0.1
            )
            
            # Max tokens
            max_tokens = st.number_input(
                "Max tokens",
                min_value=100,
                max_value=4000,
                value=st.session_state.settings.get("max_tokens", 2000),
                step=100
            )
            
            # Research i bibliografia
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                do_research = st.checkbox(
                    "Przeprowadź research",
                    value=st.session_state.settings.get("do_research", False)
                )
                
            with col_r2:
                add_bibliography = st.checkbox(
                    "Dodaj bibliografię",
                    value=st.session_state.settings.get("add_bibliography", True)
                )
                
            # Opcje dodatkowe dla researchu
            if do_research:
                search_context_size = st.radio(
                    "Ilość danych z wyszukiwania",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(st.session_state.settings.get("search_context_size", "medium")),
                    horizontal=True,
                    help="Ilość kontekstu wyszukiwania: Low (mniejszy koszt, szybsza odpowiedź), Medium (zbalansowane), High (najwięcej danych, wyższy koszt)"
                )
            else:
                search_context_size = "medium"
        
        # Aktualizacja ustawień
        updated_settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "do_research": do_research,
            "search_context_size": search_context_size,
            "add_bibliography": add_bibliography,
            "search_intent": search_intent,
            "complexity_level": complexity_level,
            "engagement_elements": {
                "rhetorical_questions": rhetorical_questions,
                "statistics_quotes": statistics_quotes,
                "examples_cases": examples_cases,
                "stories": stories
            },
            "content_type": content_type,
            "readability_index": readability_index,
            "num_topics": num_topics,
            "num_sections": num_sections,
            "section_length": section_length
        }
        
        # Zapisywanie ustawień jeśli się zmieniły
        if updated_settings != st.session_state.settings:
            st.session_state.settings = updated_settings
            save_settings(updated_settings)
            
        # Słowa kluczowe
        keywords = st.text_input(
            "Wprowadź słowa kluczowe (oddzielone przecinkami):"
        )
        
        # Przyciski akcji
        if st.button("Generuj tematy") and keywords:
            with st.spinner("Generowanie tematów..."):
                st.session_state.topics = st.session_state.generator.generate_topics(keywords, st.session_state.settings)
    
    # Panel wyników (prawa kolumna)
    with col2:
        st.header("Panel wyników")
        
        # Wyświetlanie tematów
        if st.session_state.topics:
            st.subheader("Propozycje tematów")
            
            # Przygotowanie danych do tabeli
            topics_df = []
            for topic in st.session_state.topics:
                topics_df.append({
                    "ID": topic["id"],
                    "Tytuł": topic["title"],
                    "Słowa kluczowe": ", ".join(topic["keywords"]),
                    "Opis": topic["description"]
                })
                
            # Wyświetlenie tabeli
            st.dataframe(topics_df)
            
            # Wybór tematu
            topic_options = [f"{t['id']}. {t['title']}" for t in st.session_state.topics]
            selected_topic = st.selectbox(
                "Wybierz temat do wygenerowania:",
                options=topic_options
            )
            
            if selected_topic:
                selected_id = int(selected_topic.split('.')[0])
                st.session_state.selected_topic_index = next(
                    (i for i, t in enumerate(st.session_state.topics) if t["id"] == selected_id), 
                    None
                )
                
                # Jeśli wybrano temat
                if st.session_state.selected_topic_index is not None:
                    topic = st.session_state.topics[st.session_state.selected_topic_index]
                    
                    # Przeprowadzenie researchu (jeśli włączone)
                    if st.session_state.settings.get("do_research", False):
                        if st.button("Przeprowadź research"):
                            with st.spinner("Przeprowadzanie researchu..."):
                                research_content = st.session_state.generator.perform_web_research(topic, st.session_state.settings)
                                if research_content:
                                    st.success("Research przeprowadzony pomyślnie")
                                    st.markdown("### Wyniki researchu")
                                    st.write(research_content)
                    
                    # Generowanie konspektu
                    if st.button("Generuj konspekt"):
                        with st.spinner("Generowanie konspektu artykułu..."):
                            outline = st.session_state.generator.generate_article_outline(topic, st.session_state.settings)
                            if outline:
                                st.session_state.outline = outline
                                st.session_state.article_content = None  # Resetujemy treść artykułu
                    
                    # Wyświetlanie konspektu
                    if st.session_state.outline:
                        st.subheader("Konspekt artykułu")
                        st.json(st.session_state.outline)
                        
                        # Generowanie treści
                        if st.button("Generuj treść artykułu"):
                            with st.spinner("Generowanie artykułu..."):
                                # Użycie nowej funkcji do generowania artykułu sekcja po sekcji
                                html_content = st.session_state.generator.generate_article_section_by_section(
                                    st.session_state.outline,
                                    topic,
                                    st.session_state.settings
                                )
                                
                                st.session_state.article_content = {
                                    "text": html_content,
                                    "html": html_content
                                }
                                
                                st.success("Artykuł wygenerowany!")
                        
                        # Wyświetlanie artykułu
                        if st.session_state.article_content:
                            st.subheader("Wygenerowany artykuł")
                            
                            # Przełącznik widoku
                            view_options = st.radio(
                                "Widok",
                                options=["Podgląd", "Źródło HTML"],
                                horizontal=True,
                                index=0 if st.session_state.html_view == "podgląd" else 1
                            )
                            
                            st.session_state.html_view = "podgląd" if view_options == "Podgląd" else "źródło"
                            
                            # Wyświetlanie w zależności od wybranego widoku
                            if st.session_state.html_view == "podgląd":
                                st.components.v1.html("""
                                <style>
                                body {
                                    background-color: white;
                                    color: black;
                                    font-family: Arial, sans-serif;
                                    margin: 20px;
                                    line-height: 1.6;
                                }
                                h3 {
                                    color: #333;
                                    border-bottom: 1px solid #ddd;
                                    padding-bottom: 5px;
                                }
                                p {
                                    margin-bottom: 15px;
                                }
                                ul, ol {
                                    margin-bottom: 15px;
                                    padding-left: 20px;
                                }
                                li {
                                    margin-bottom: 5px;
                                }
                                strong {
                                    color: #111;
                                }
                                a {
                                    color: #0066cc;
                                    text-decoration: none;
                                }
                                a:hover {
                                    text-decoration: underline;
                                }
                                </style>
                                """ + st.session_state.article_content["html"], height=600, scrolling=True)
                            else:
                                st.code(st.session_state.article_content["html"], language="html")
                                
                            # Przycisk do kopiowania
                            st.markdown(get_copy_to_clipboard_button(st.session_state.article_content["html"]), unsafe_allow_html=True)
                            
                            # Przycisk do pobrania jako HTML
                            st.download_button(
                                label="Pobierz jako HTML",
                                data=st.session_state.article_content["html"],
                                file_name=f"{st.session_state.outline['article_title'].replace(' ', '_')}.html",
                                mime="text/html"
                            )

if __name__ == "__main__":
    main()