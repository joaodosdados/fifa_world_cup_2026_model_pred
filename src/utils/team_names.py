"""
Team name mappings - Full names to TV abbreviations
"""

# Mapping of full team names to TV-style abbreviations
TEAM_ABBREVIATIONS = {
    # Americas
    'Argentina': 'ARG',
    'Brazil': 'BRA',
    'Canada': 'CAN',
    'Chile': 'CHI',
    'Colombia': 'COL',
    'Costa Rica': 'CRC',
    'Ecuador': 'ECU',
    'Honduras': 'HON',
    'Jamaica': 'JAM',
    'Mexico': 'MEX',
    'Panama': 'PAN',
    'Paraguay': 'PAR',
    'Peru': 'PER',
    'Trinidad and Tobago': 'TRI',
    'United States': 'USA',
    'USA': 'USA',
    'Uruguay': 'URU',
    'Venezuela': 'VEN',
    
    # Europe
    'Austria': 'AUT',
    'Belgium': 'BEL',
    'Bosnia and Herzegovina': 'BIH',
    'Bulgaria': 'BUL',
    'Croatia': 'CRO',
    'Czech Republic': 'CZE',
    'Denmark': 'DEN',
    'England': 'ENG',
    'France': 'FRA',
    'Germany': 'GER',
    'Greece': 'GRE',
    'Hungary': 'HUN',
    'Iceland': 'ISL',
    'Italy': 'ITA',
    'Netherlands': 'NED',
    'Northern Ireland': 'NIR',
    'Norway': 'NOR',
    'Poland': 'POL',
    'Portugal': 'POR',
    'Republic of Ireland': 'IRL',
    'Romania': 'ROU',
    'Russia': 'RUS',
    'Scotland': 'SCO',
    'Serbia': 'SRB',
    'Slovakia': 'SVK',
    'Slovenia': 'SVN',
    'Spain': 'ESP',
    'Sweden': 'SWE',
    'Switzerland': 'SUI',
    'Turkey': 'TUR',
    'Ukraine': 'UKR',
    'Wales': 'WAL',
    
    # Asia
    'Australia': 'AUS',
    'China PR': 'CHN',
    'Iran': 'IRN',
    'Iraq': 'IRQ',
    'Japan': 'JPN',
    'Korea Republic': 'KOR',
    'República da Coreia': 'KOR',  # Portuguese name
    'Korea DPR': 'PRK',
    'Saudi Arabia': 'KSA',
    'South Korea': 'KOR',
    'United Arab Emirates': 'UAE',
    
    # Africa
    'Algeria': 'ALG',
    'Cameroon': 'CMR',
    'Egypt': 'EGY',
    'Ghana': 'GHA',
    'Ivory Coast': 'CIV',
    "Côte d'Ivoire": 'CIV',
    'Morocco': 'MAR',
    'Nigeria': 'NGA',
    'Senegal': 'SEN',
    'South Africa': 'RSA',
    'Tunisia': 'TUN',
    
    # Oceania
    'New Zealand': 'NZL',
}

PREDICTION_TEAM_NAMES = {
    'Alemanha': 'Germany',
    'Argélia': 'Algeria',
    'Arábia Saudita': 'Saudi Arabia',
    'Austrália': 'Australia',
    'Áustria': 'Austria',
    'Bélgica': 'Belgium',
    'Bósnia e Herzegovina': 'Bosnia and Herzegovina',
    'Brasil': 'Brazil',
    'Canadá': 'Canada',
    'Catar': 'Qatar',
    'Colômbia': 'Colombia',
    'Costa do Marfim': 'Ivory Coast',
    'Croácia': 'Croatia',
    'Egito': 'Egypt',
    'Equador': 'Ecuador',
    'Escócia': 'Scotland',
    'Espanha': 'Spain',
    'EUA': 'USA',
    'França': 'France',
    'Gana': 'Ghana',
    'Holanda': 'Netherlands',
    'Inglaterra': 'England',
    'Iraque': 'Iraq',
    'Japão': 'Japan',
    'Marrocos': 'Morocco',
    'México': 'Mexico',
    'Noruega': 'Norway',
    'Nova Zelândia': 'New Zealand',
    'Panamá': 'Panama',
    'Paraguai': 'Paraguay',
    'RD do Congo': 'Zaire',
    'RI do Irã': 'Iran',
    'República da Coreia': 'Korea Republic',
    'Senegal': 'Senegal',
    'Suécia': 'Sweden',
    'Suíça': 'Switzerland',
    'Tchéquia': 'Czech Republic',
    'Tunísia': 'Tunisia',
    'Turquia': 'Turkey',
    'Uruguai': 'Uruguay',
    'África do Sul': 'South Africa',
}


def normalize_team_name(team_name: str) -> str:
    """Map display names to the identifiers used by historical models."""
    clean_name = str(team_name).strip()
    return PREDICTION_TEAM_NAMES.get(clean_name, clean_name)

def get_team_abbreviation(team_name):
    """
    Get TV-style abbreviation for a team name.
    
    Parameters:
    -----------
    team_name : str
        Full team name
    
    Returns:
    --------
    str
        3-letter abbreviation (e.g., 'BRA', 'ARG', 'KOR')
    """
    return TEAM_ABBREVIATIONS.get(team_name, team_name[:3].upper())

def get_team_abbreviation_with_tooltip(team_name):
    """
    Get TV-style abbreviation with HTML tooltip showing full name.
    
    Parameters:
    -----------
    team_name : str
        Full team name
    
    Returns:
    --------
    str
        HTML string with abbreviation and tooltip
    """
    abbr = get_team_abbreviation(team_name)
    return f'<span title="{team_name}" style="cursor: help; border-bottom: 1px dotted #666;">{abbr}</span>'

def get_team_display_name(team_name, use_abbreviation=True):
    """
    Get display name for a team (abbreviation or full name).
    
    Parameters:
    -----------
    team_name : str
        Full team name
    use_abbreviation : bool
        If True, return abbreviation; if False, return full name
    
    Returns:
    --------
    str
        Display name
    """
    if use_abbreviation:
        return get_team_abbreviation(team_name)
    return team_name

# Made with Bob
