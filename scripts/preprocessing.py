import numpy as np
import pandas as pd 
import os
from pathlib import Path

def preprocess(df):
    
    df = df.copy()

    White = ['White Blend', 'Pinot Gris', 'Riesling', 'Chardonnay', 'Chenin Blanc', 'Sauvignon Blanc', 
            'Viognier-Chardonnay', 'Catarratto', 'Inzolia', 'Bordeaux-style White Blend', 'Grillo', 
            'Albariño', 'Petit Manseng', 'Vernaccia', 'Grüner Veltliner', 'Viognier', 'Vermentino', 
            'Grenache Blanc', 'Pinot Blanc', 'Alsace white blend', 'Portuguese White', 'Verdejo', 
            'Fumé Blanc', 'Pinot Bianco', 'Ugni Blanc-Colombard', 'Friulano', 'Assyrtico', 'Vignoles', 
            'Muscat', 'Muscadelle', 'Garganega', 'Pinot Grigio','Cortese', 'Melon', 'Vidal', 'Verdelho', 
            'Marsanne', 'Vilana', 'Viura', 'Verduzzo', 'Verdicchio', 'Colombard', 'Sylvaner', 'Sémillon', 
            'Antão Vaz', 'Verdejo-Viura', 'Chenin Blanc-Chardonnay', 'Insolia', 'Ribolla Gialla', 
            'Weissburgunder', 'Traminer', 'Prié Blanc', 'Müller-Thurgau', 'Pansa Blanca', 'Muskat Ottonel',
            'Sauvignon Blanc-Semillon', 'Semillon-Sauvignon Blanc', 'Bical', 'Viura-Chardonnay', 'Malvasia Bianca',
            'Rhône-style White Blend', 'Scheurebe', 'Kerner', 'Carricante', 'Fiano', 'Früburgunder', 'Roussanne', 
            'Avesso', 'Chinuri', 'Muscat Blanc à Petits Grains', 'Xarel-lo', 'Greco', 'Trebbiano', 'Prié Blanc',
            'Falanghina', 'Bical', 'Gelber Muskateller', 'Turbiana', 'Refosco', 'Alvarinho', 'Manzoni', 'Assyrtiko', 
            'Welschriesling', 'Rieslaner', 'Traminette', 'Marsanne-Viognier', 'Gewürztraminer-Riesling', 
            'Austrian white blend', 'Tocai', 'Chardonnay-Viognier', 'Fernão Pires', 'Seyval Blanc', 'Muscat Canelli', 
            'Arinto', 'Arneis', 'Malvasia', 'Altesse', 'Blanc du Bois', 'Provence white blend', 'Nosiola', 
            'Roussanne-Viognier', 'Godello', 'Auxerrois', 'Albana', 'Muskat',  'Grechetto', 'Encruzado', 
            'Garnacha Blanca', 'Pallagrello', 'Morava', 'Aleatico', 'Nascetta', 'Siria', 'Asprinio', 'Feteascǎ Regalǎ', 
            'Tocai Friulano', 'Schiava', 'Chardonnay-Semillon', 'Palomino', 'Norton', 
            'Loureiro-Arinto', 'Symphony', 'Edelzwicker', 'Madeira Blend', 'Gros and Petit Manseng', 'Jacquère', 
            'Chenin Blanc-Sauvignon Blanc', 'Marzemino', 'Chardonnay-Sauvignon Blanc', 'Trebbiano Spoletino',
            'Chasselas', 'Hárslevelü', 'Siegerrebe','Colombard-Sauvignon Blanc', 'Diamond',
            'Gros Manseng', 'Muskateller', 'Aligoté', 'Muscat Blanc', 'Viognier-Roussanne', 'Pallagrello Bianco', 
            'Veltliner', 'Chardonnay-Sauvignon', 'Chenin Blanc-Viognier', 'Vitovska', 'Grauburgunder', 'Macabeo', 
            'Verdil', 'Treixadura', 'Coda di Volpe', 'Viura-Verdejo', 'Bombino Bianco', 'Pinot-Chardonnay', 
            "Muscat d'Alexandrie", 'Chardonnay-Pinot Gris', 'Chardonnay-Pinot Blanc','Piquepoul Blanc', 'Orange Muscat',
            'Ugni Blanc', 'Semillon-Chardonnay', 'Irsai Oliver', 'Greco Bianco', 'Viognier-Grenache Blanc', 'Pignoletto', 
            'Muscatel', 'White Riesling', 'Hondarrabi Zuri', 'Nuragus', 'Xynisteri', 'Sauvignon Musqué', 'Roussanne-Marsanne', 
            'Incrocio Manzoni', 'Terrantez', 'Bual', 'Verdejo-Sauvignon Blanc', 'Malvasia-Viura', 'Savatiano', 
            'Macabeo-Chardonnay', 'Tamjanika', 'Macabeo-Moscatel', 'Códega do Larinho','Pinot Gris-Gewürztraminer',
            'Viosinho', 'Paralleda', 'Malvar', 'Airen', 'Erbaluce', 'Verdosilla', 'Aidani', 'Vinhão', 'Rolle', 'Orangetraube', 
            'Žilavka', 'Portuguiser', 'Gouveio', 'Bombino Nero', 'Malagouzia-Chardonnay', 'Elbling', 'Gragnano', 
            'Pinot Blanc-Chardonnay', 'Petit Meslier', 'Chardonnay Weissburgunder', 'Robola', 'Folle Blanche', 'Malagouzia', 
            'Rabigato', 'Sauvignonasse', 'Meseguera', 'Alvarinho-Chardonnay', 'Pinot Blanc-Viognier', 'Biancu Gentile', 
            'Xinisteri','Moschofilero-Chardonnay','Sauvignon Blanc-Sauvignon Gris', 'Trebbiano di Lugana', 'Verdeca', 
            'Chardonel', 'Silvaner-Traminer', 'Uvalino', 'Merseguera-Sauvignon Blanc', 'Cayuga', 
            'Nasco', 'Vital', 'Apple', 'Pinot Grigio-Sauvignon Blanc', 'Valvin Muscat', 'Malvasia Fina', 
            'Roditis-Moschofilero', 'Premsal', 'Jampal', 'Tokay Pinot Gris', 'Trajadura', 'Roscetto', 'Torontel', 
            'Viognier-Valdiguié',
            'Zierfandler', 'Marsanne-Roussanne', 'Pinot Meunier', 'Muskat Ottonel', 'Moscatel', 'Moschofilero', 'White Port', 
            'Kisi', 'Kangoun', 'Posip', 'Uva di Troia', 'Zierfandler-Rotgipfler', 'Mauzac', 'Pinot Auxerrois', 'Neuburger', 
            'Sämling', 'Rkatsiteli', 'Trousseau Gris', 'Malvasia Istriana', 'Morillon', 'Tokay', 'Gros Plant', 'Muscat Hamburg', 
            'Emir', 'Tsolikouri', 'Narince', 'Grecanico', 'Madeleine Angevine', 'Doña Blanca', 'Graševina', 'Thrapsathiri', 
            'Cococciola', 'Plyto', 'Azal', 'Moscatel Graúdo', 'Malvasia di Candia', 'Maria Gomes', 'Muscat of Alexandria', 
            'Moscatel de Alejandría', 'Misket', 'Tamianka', 'Morio Muskat', 'Sauvignonasse', 
            'Viognier-Marsanne', 'Ryzlink Rýnský', 'Muscadel', 'Roussanne-Grenache Blanc', 'Chancellor', 'Picapoll', 
            'Blauburger', 'Athiri', 'Ondenc','Gewürztraminer', 'Torrontés', 'Furmint', 'Savagnin', 'Glera', 
            'Roter Veltliner', 'Silvaner', 'Ruché', 'Pecorino', 'Sauvignon Gris', 'Vidal Blanc', 'Albanello', 
            'Loureiro', 'Clairette', 'Verduzzo Friulano ', "Loin de l'Oeil", 'Timorasso', 'Pigato', 'Viognier-Gewürztraminer', 
            'Sauvignon Blanc-Chenin Blanc', 'Colombard-Ugni Blanc', 'Mtsvane', 'Rivaner', 'Vespaiolo', 'Biancolella', 
            'Riesling-Chardonnay', 'Maria Gomes-Bical', 'Gelber Traminer', 'Sercial', 'Grenache Gris', 'Chardonnay-Albariño',
            'Roditis', 'Papaskarasi', 'Zibibbo', 'Malagousia', 'Rotgipfler', 'Durella', 'Cercial', 'Johannisberg Riesling', 
            'Teran', 'Mantonico', 'Timorasso', 'Zlahtina', 'Shiraz-Roussanne', 'Tămâioasă Românească', 'Ansonica', 'Feteasca',
            'Catalanesca', 'Moscato di Noto', 'Moscato Giallo','Sauvignon Blanc-Chardonnay', 'Sauvignon-Sémillon', "Cesanese d'Affile", 
            'Sauvignon Blanc-Verdejo', 'Chardonnay-Riesling', 'Sauvignon Blanc-Assyrtiko','Zelen', 'Tempranillo Blanco', 
            'Roter Traminer','Picpoul'
    ]
    Red = ['Portuguese Red', 'Pinot Noir', 'Tempranillo-Merlot', 'Frappato', 'Cabernet Sauvignon',
            'Nerello Mascalese', 'Malbec', 'Tempranillo Blend', 'Meritage', 'Red Blend', 'Merlot', 
            "Nero d'Avola", 'Gamay', 'Primitivo', 'Sangiovese', 'Cabernet Franc', 'Bordeaux-style Red Blend', 
            'Aglianico', 'Petite Sirah', 'Touriga Nacional', 'Carmenère', 'Rosso', 'Shiraz-Cabernet Sauvignon', 
            'Barbera', 'Rhône-style Red Blend', 'Graciano', 'Tannat-Cabernet', 'Sauvignon', 'Sangiovese Grosso', 
            'Bonarda', 'Shiraz', 'Montepulciano', 'Grenache', 'Syrah', 'Nebbiolo', 'Blaufränkisch', 'Carignan-Grenache', 
            'Sagrantino', 'Cabernet Sauvignon-Syrah', 'Tempranillo','Mencía', 'Zweigelt', 'Cannonau', 'Dolcetto', 
            'Garnacha Tintorera', 'Pinot Nero', 'Pinotage', 'Syrah-Grenache', 'Antão Vaz', 'Cabernet Sauvignon-Carmenère', 
            'Tinta Miúda', 'Monastrell', 'Merlot-Malbec', 'Cabernet Sauvignon-Merlot', 'Merlot-Argaman', 'Garnacha', 
            'Negroamaro', 'Mourvèdre', 'Syrah-Cabernet', 'Tannat', 'Cabernet Sauvignon-Sangiovese', 'Austrian Red Blend', 
            'Teroldego', 'Baga','Pinot Noir-Gamay', 'Cinsault', 'Corvina, Rondinella, Molinara', 'Tannat-Syrah', 'Charbono', 
            'Provence red blend', 'Claret','Malbec-Merlot', 'Monastrell-Syrah', 'Malbec-Tannat', 'Malbec-Cabernet Franc', 
            'Tinta de Toro', 'Cabernet Moravia', 'Chambourcin', 'Nero di Troia', 'Cesanese', 'Lagrein', 'Tinta Fina', 'St. Laurent', 
            'Cabernet Sauvignon-Shiraz', 'Syrah-Cabernet Sauvignon', 'Pugnitello', 'Touriga Nacional Blend', 'Tinta Roriz', 
            'Cabernet Franc-Cabernet Sauvignon', 'Grenache-Syrah', 'Tempranillo-Cabernet Sauvignon', 'Merlot-Cabernet Franc', 
            'Syrah-Petite Sirah', 'Cabernet Blend', 'Maturana', 'Magliocco', 'Gamay Noir', 'Spätburgunder', 'Plavac Mali',
            'Lemberger', 'Saperavi', 'Dornfelder', 'Ojaleshi', 'Mondeuse', 'Perricone', 'Syrah-Merlot', 'Cabernet Sauvignon-Malbec',
            'Tinto Fino', 'Malbec-Cabernet Sauvignon','Carignano', 'Cabernet Franc-Merlot', 
            'Syrah-Petit Verdot', 'Syrah-Mourvèdre', 'Shiraz-Grenache', 'Grenache-Carignan', 'Malbec-Syrah', 
            'Cabernet Sauvignon-Tempranillo', 'Carignan', 'Cabernet-Syrah', 'Merlot-Cabernet Sauvignon', 
            'Mourvèdre-Syrah', 'Negrette', 'Tinta Barroca', 'Merlot-Tannat','Castelão', 
            'Grenache Blend', 'Sangiovese Cabernet', 'Touriga Nacional-Cabernet Sauvignon', 'Cabernet Sauvignon-Cabernet Franc', 
            'Baco Noir', 'Tempranillo-Tannat', 'Touriga Franca', 'Barbera-Nebbiolo', 'Prieto Picudo', 'Gaglioppo', 'Carignane', 
            'Tannat-Merlot', 'Nerello Cappuccio', 'Counoise', 'Mazuelo', 'Tinta del Pais', 'Vranec', 'Mavrud', 'Cabernet', 
            'Grenache-Mourvèdre', 'Forcallà', 'Syrah-Tempranillo', 'Cabernet Sauvignon-Barbera', 'Merlot-Cabernet', 'Jaen', 
            'Tinta del Toro', 'Prunelard', 'Garnacha-Syrah', 'Rufete', 'Tempranillo-Shiraz','Mansois',
            'Mataro', 'Tinta Cao', 'Blauer Portugieser', 'Groppello', 'Poulsard', 'Grenache-Shiraz', 'Baga-Touriga Nacional', 
            'Carineña', 'Ciliegiolo', 'Cabernet Sauvignon-Merlot-Shiraz', 'Sciaccerellu', 'Alicante', 'Rosenmuskateller', 
            'Malbec-Cabernet', 'Touriga', 'Carmenère-Syrah', 'Mavroudi', 'Pinot Blanc-Pinot Noir', 'Tinto Velasco', 'Kadarka', 
            'Sangiovese-Syrah', 'Tannat-Cabernet Franc', 'Fer Servadou', 'Mission', 'Kekfrankos', 'Blauburgunder', 'Marquette', 
            'Romorantin', 'Braucol', 'Cabernet Franc-Malbec', 'Pallagrello Nero', 'Rebula', 'Vespolina', 'Shiraz-Malbec', 
            'Rebo', 'Tempranillo-Malbec', 'Trousseau', 'Bacchus', 'Syrah-Malbec', 'Syrah-Cabernet Franc', 'Cariñena-Garnacha', 
            'Sideritis','Rara Neagra', 'Molinara', 'Abouriou', 'Nielluciu', 'Malbec-Bonarda', 'Garnacha-Monastrell', 'Souzao', 
            'Tinta Francisca', 'Malvasia Nera', 'Listán Negro', 'Pinotage-Merlot', 'Jacquez', 'Carignan-Syrah', 'Mavrotragano', 
            'Bovale', 'Frankovka', 'Garnacha Blend', 'Merlot-Shiraz', 'Malbec Blend', 'Merlot-Syrah', 'Babić', 'Yapincak', 
            'Mandilaria', 'Saperavi-Merlot', 'Teroldego Rotaliano', 'Garnacha-Tempranillo','Vermentino Nero',
            'Albarossa', 'Cabernet Sauvignon Grenache', 'Black Monukka', 'Merlot-Grenache', 'Vranac', 'Tempranillo-Syrah', 
            'Boğazkere', 'Tinta Amarela', 'Tinta Negra Mole', 'Chelois', 'Shiraz-Tempranillo', 'Biancale', 'Syrah-Bonarda', 
            'Durif', 'Franconia', 'Malbec-Tempranillo', 'Monastrell-Petit Verdot', 'Sirica', 'Espadeiro', 'Blatina', 'Karalahna', 
            'Garnacha-Cabernet', 'Garnacha-Cariñena', 'Cabernet Franc-Lemberger', 'Shiraz-Mourvèdre', 'Mavrokalavryta', 'Favorita', 
            'Babosa Negro', 'Dafni', 'Petit Courbu', 'Kotsifali', 'Parraleta', 'Otskhanuri Sapere', 'Trollinger', 
            'Tsapournakos', 'Francisa', 'Kuntra', 'Pignolo', 'Schwartzriesling','Sousão', 'Feteasca Neagra', 'Kinali Yapincak',
            'Kalecik Karasi', 'Karasakiz', 'Raboso', 'Trepat', 'Freisa', 'Trincadeira', 'Melnik', 'Argaman', 'Piedirosso', 
            'Marawi', 'Çalkarası', 'Tinta Francisca', 'Vidadillo', 'Other', 'Cabernet Pfeffer', 'Roviello', 'Colorino', 
            'Tinta Madeira', 'Centesimino', 'Ramisco', 'Gamza', 'Bobal-Cabernet Sauvignon',
            'Petit Verdot', 'Zinfandel', 'G-S-M', 'Monica', 'Cabernet Merlot', 'Cabernet Franc-Carmenère', 
            'Grenache Noir', 'Xinomavro', 'Petite Verdot', 'Tempranillo-Garnacha', 'Carmenère-Cabernet Sauvignon', 
            'Sangiovese-Cabernet Sauvignon', 'Shiraz-Cabernet', 'Syrah-Grenache-Viognier', 'Cabernet-Shiraz', 'Syrah-Carignan', 
            'Cabernet-Malbec', 'Merlot-Petite Verdot', 'Duras', 'Aragonês', 'Agiorgitiko', 'Aragonez', 'Alfrocheiro', 'Corvina', 
            'Alicante Bouschet', 'Tinto del Pais', 'Bobal', 'Susumaniello', 'Grolleau', 'Canaiolo', 'Bastardo', 'Tintilia', 
            'St. Vincent', 'Caprettone','Black Muscat','Muscadine','Syrah-Viognier', 'Shiraz-Viognier', 'Carcajolu', 
            'Marselan', 'Malbec-Petit Verdot', 'Grignolino', 'Pinot Noir-Syrah', 'Malbec-Carménère','País', 'Alvarelhão', 
            'Okuzgozu', 'Tintilia','Mavrodaphne','Tintilia ', 
    ] 

    Rosé = ['Rosé', 'Rosato', 'Rosado','Portuguese Rosé', 'Prugnolo Gentile'] 
    
    Sparkling = ['Champagne Blend', 'Prosecco', 'Sparkling Blend','Portuguese Sparkling',
                'Cerceal', 'Lambrusco','Lambrusco di Sorbara','Lambrusco Grasparossa',
                'Torbato', 'Moscadello', 'Passerina', 'Brachetto', 'Ekigaïna', 'Picolit', 
                'Sacy', 'Moscatel Roxo', 'Debit','Moscato', 'Valdiguié', 'Casavecchia', 
                'Lambrusco Salamino', 'Moscato Rosa'] 
    
    Fortified = ['Sherry', 'Pedro Ximénez', 'White Port', 'Tokaji','Port']

    red_dict = {variety: 'Red' for variety in Red}
    white_dict = {variety: 'White' for variety in White}
    rose_dict = {variety: 'Rosé' for variety in Rosé}
    sparkling_dict = {variety: 'Sparkling' for variety in Sparkling}
    fortified_dict = {variety: 'Fortified' for variety in Fortified}
    wine_dict = {**red_dict, **white_dict, **rose_dict, **sparkling_dict, **fortified_dict}
    
    # Remove duplicates 
    df = df.drop_duplicates(subset='description', keep="first")
    
    # Apply wine type dict map 
    df['type'] = df['variety'].map(wine_dict)

    # Fix one missing value:
    df['type'].fillna('Red', inplace=True)
        
    # Rename cols
    df = df.rename(columns={'country':'Country',
        'description':'Tasting notes',
        'designation':'Designation',
        'points':'Score', 
        'price': 'Price',
        'province':'Province',
        'region_1': 'Region', 
        'title':'Title', 
        'variety':'Variety',
        'winery':'Winery',
        'embeddings':'embeddings', 
        'type':'Type'}) 
    
    return df