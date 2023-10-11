from pycldf import Dataset

nelex = Dataset.from_metadata("data/northeuralex-0.9/cldf-metadata.json")


def get_segments_by_language():
    forms = nelex.get('FormTable')
    segments_by_lang = {}

    for row in nelex.iter_rows(forms):
        lang = row.get('Language_ID')
        segments = row.get('Segments')

        if lang in segments_by_lang:
            segments_by_lang[lang].append(segments)
        else:
            segments_by_lang[lang] = [segments]

    return segments_by_lang
