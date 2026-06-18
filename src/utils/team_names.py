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
