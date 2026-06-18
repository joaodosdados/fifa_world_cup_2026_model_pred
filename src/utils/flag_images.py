"""
Flag image utilities for country flags
Uses flagcdn.com API for high-quality flag images
"""

# Mapping of country names to ISO 3166-1 alpha-2 codes
COUNTRY_CODES = {
    # Group A
    'México': 'mx',
    'República da Coreia': 'kr',
    'Tchéquia': 'cz',
    'África do Sul': 'za',
    
    # Group B
    'Suíça': 'ch',
    'Canadá': 'ca',
    'Catar': 'qa',
    'Bósnia e Herzegovina': 'ba',
    
    # Group C
    'Escócia': 'gb-sct',  # Scotland
    'Marrocos': 'ma',
    'Brasil': 'br',
    'Haiti': 'ht',
    
    # Group D
    'EUA': 'us',
    'Austrália': 'au',
    'Turquia': 'tr',
    'Paraguai': 'py',
    
    # Group E
    'Alemanha': 'de',
    'Costa do Marfim': 'ci',
    'Equador': 'ec',
    'Curaçau': 'cw',
    
    # Group F
    'Suécia': 'se',
    'Japão': 'jp',
    'Holanda': 'nl',
    'Tunísia': 'tn',
    
    # Group G
    'Nova Zelândia': 'nz',
    'RI do Irã': 'ir',
    'Bélgica': 'be',
    'Egito': 'eg',
    
    # Group H
    'Uruguai': 'uy',
    'Arábia Saudita': 'sa',
    'Espanha': 'es',
    'Cabo Verde': 'cv',
    
    # Group I
    'Noruega': 'no',
    'França': 'fr',
    'Senegal': 'sn',
    'Iraque': 'iq',
    
    # Group J
    'Argentina': 'ar',
    'Áustria': 'at',
    'Jordânia': 'jo',
    'Argélia': 'dz',
    
    # Group K
    'Colômbia': 'co',
    'RD do Congo': 'cd',
    'Portugal': 'pt',
    'Uzbequistão': 'uz',
    
    # Group L
    'Inglaterra': 'gb-eng',  # England
    'Gana': 'gh',
    'Panamá': 'pa',
    'Croácia': 'hr',
}

def get_flag_url(country_name: str, size: str = 'w40') -> str:
    """
    Get flag image URL from flagcdn.com
    
    Args:
        country_name: Name of the country
        size: Size of the flag (w20, w40, w80, w160, w320, w640, w1280, w2560)
    
    Returns:
        URL to the flag image
    """
    code = COUNTRY_CODES.get(country_name, 'xx')
    return f"https://flagcdn.com/{size}/{code}.png"

def get_flag_html(country_name: str, size: str = 'w40', height: int = 20) -> str:
    """
    Get HTML img tag for flag
    
    Args:
        country_name: Name of the country
        size: Size of the flag from API
        height: Height in pixels for display
    
    Returns:
        HTML img tag
    """
    url = get_flag_url(country_name, size)
    return f'<img src="{url}" height="{height}" style="vertical-align: middle; margin-right: 8px;">'

def get_country_code(country_name: str) -> str:
    """Get ISO country code for a country name"""
    return COUNTRY_CODES.get(country_name, 'xx')

# Made with Bob
