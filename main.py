import yaml

from os.path import join, dirname, abspath, isdir
from os import makedirs

from time import strftime
from resite.resite import Resite
from iepy.geographics.codes import get_subregions

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

params = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

if __name__ == '__main__':

    output_folder = join(dirname(abspath(__file__)), f"output/{strftime('%Y%m%d_%H%M%S')}/")
    # Compute and save results
    if not isdir(output_folder):
        makedirs(output_folder)

    logger.info('Building class.')
    countries = get_subregions(params['regions'])
    resite = Resite(countries, params["technologies"], params["timeslice"],
                    params["spatial_resolution"], params['min_cap_if_selected'])

    logger.info('Reading input.')
    resite.build_data(use_ex_cap=params["use_ex_cap"], min_cap_pot=params["min_cap_pot"])

    logger.info('Model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['formulation_params'],
                       params['write_lp'], output_folder)

    logger.info('Sending model to solver.')
    results = resite.solve_model(output_folder=output_folder,
                                 solver_options=params['solver_options'], solver=params["solver"])
    logger.info('Retrieving results.')
    resite.retrieve_selected_sites_data()

    resite.save(output_folder)
