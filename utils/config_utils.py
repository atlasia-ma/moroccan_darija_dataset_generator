import configparser
def get_config(key, header):
    if header is None :
        header = "DEFAULT"
    config = configparser.ConfigParser()
    config.read('./config/config.properties')
    return config[header][key]