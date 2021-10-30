from transforms.api import transform, Input


@transform(
     cs=Input('/N3C Export Area/Concept Set Ontology/Concept Set Ontology/hubble_base/concept_set_members'),
     # processed=Output('/examples/hair_eye_color_processed')
)
def concept_set_members():
    filtered_df = cs.dataframe()
    return filtered_df
