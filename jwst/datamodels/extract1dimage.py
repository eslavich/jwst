from .model_base import DataModel

__all__ = ['Extract1dImageModel']

class Extract1dImageModel(DataModel):
    """
    A data model for the extract_1d reference image array.

    Parameters
    __________
    data : numpy float32 array
         1-D extraction regions array
    """
    schema_uri = "http://stsci.edu/schemas/jwst_datamodel/extract1dimage.schema"
