from pydantic import BaseModel


# to specify config type of elements witch will be in the setting of API,
# witch will also be the first get endpoint of api
class SettingsGetEndpoint(BaseModel):
    name: str
    api_version: str
    model_version: str

    # to specify that there are no protected namespaces for class SettingsGetEndpoint,
    # which should resolve the conflict with the "model_version" field in pydantic library
    class Config:
        protected_namespaces = ()
