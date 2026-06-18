"""
Country flag emojis for World Cup teams
"""

COUNTRY_FLAGS = {
    # Americas
    'Argentina': '🇦🇷',
    'Brazil': '🇧🇷',
    'Canada': '🇨🇦',
    'Chile': '🇨🇱',
    'Colombia': '🇨🇴',
    'Costa Rica': '🇨🇷',
    'Ecuador': '🇪🇨',
    'Honduras': '🇭🇳',
    'Jamaica': '🇯🇲',
    'Mexico': '🇲🇽',
    'Panama': '🇵🇦',
    'Paraguay': '🇵🇾',
    'Peru': '🇵🇪',
    'Trinidad and Tobago': '🇹🇹',
    'United States': '🇺🇸',
    'USA': '🇺🇸',
    'Uruguay': '🇺🇾',
    'Venezuela': '🇻🇪',
    
    # Europe
    'Austria': '🇦🇹',
    'Belgium': '🇧🇪',
    'Bosnia and Herzegovina': '🇧🇦',
    'Bulgaria': '🇧🇬',
    'Croatia': '🇭🇷',
    'Czech Republic': '🇨🇿',
    'Czechoslovakia': '🇨🇿',
    'Denmark': '🇩🇰',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'France': '🇫🇷',
    'Germany': '🇩🇪',
    'Germany FR': '🇩🇪',
    'German DR': '🇩🇪',
    'Greece': '🇬🇷',
    'Hungary': '🇭🇺',
    'Iceland': '🇮🇸',
    'Italy': '🇮🇹',
    'Netherlands': '🇳🇱',
    'Northern Ireland': '🇬🇧',
    'Norway': '🇳🇴',
    'Poland': '🇵🇱',
    'Portugal': '🇵🇹',
    'Republic of Ireland': '🇮🇪',
    'Romania': '🇷🇴',
    'Russia': '🇷🇺',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'Serbia': '🇷🇸',
    'Serbia and Montenegro': '🇷🇸',
    'Slovakia': '🇸🇰',
    'Slovenia': '🇸🇮',
    'Soviet Union': '🇷🇺',
    'Spain': '🇪🇸',
    'Sweden': '🇸🇪',
    'Switzerland': '🇨🇭',
    'Turkey': '🇹🇷',
    'Ukraine': '🇺🇦',
    'Wales': '🏴󠁧󠁢󠁷󠁬󠁳󠁿',
    'Yugoslavia': '🇷🇸',
    
    # Africa
    'Algeria': '🇩🇿',
    'Angola': '🇦🇴',
    'Cameroon': '🇨🇲',
    'Egypt': '🇪🇬',
    'Ghana': '🇬🇭',
    'Ivory Coast': '🇨🇮',
    'Morocco': '🇲🇦',
    'Nigeria': '🇳🇬',
    'Senegal': '🇸🇳',
    'South Africa': '🇿🇦',
    'Togo': '🇹🇬',
    'Tunisia': '🇹🇳',
    'Zaire': '🇨🇩',
    
    # Asia
    'Australia': '🇦🇺',
    'China PR': '🇨🇳',
    'Iran': '🇮🇷',
    'Iraq': '🇮🇶',
    'Japan': '🇯🇵',
    'Korea DPR': '🇰🇵',
    'Korea Republic': '🇰🇷',
    'Saudi Arabia': '🇸🇦',
    'United Arab Emirates': '🇦🇪',
    
    # Oceania
    'New Zealand': '🇳🇿',
    
    # Special cases
    'Dutch East Indies': '🇮🇩',
    'Bolivia': '🇧🇴',
    'Cuba': '🇨🇺',
    'El Salvador': '🇸🇻',
    'Haiti': '🇭🇹',
    'Israel': '🇮🇱',
    'Kuwait': '🇰🇼',
}

def get_flag(country_name: str) -> str:
    """
    Get flag emoji for a country
    
    Args:
        country_name: Name of the country
        
    Returns:
        Flag emoji or soccer ball if not found
    """
    return COUNTRY_FLAGS.get(country_name, '⚽')

def get_team_display_name(country_name: str) -> str:
    """
    Get team name with flag emoji
    
    Args:
        country_name: Name of the country
        
    Returns:
        Country name with flag emoji
    """
    flag = get_flag(country_name)
    return f"{flag} {country_name}"

# Made with Bob
