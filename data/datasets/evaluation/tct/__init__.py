import logging
from .tct_eval import do_tct_evaluation

def tct_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("tct.inference")
    if box_only:
        logger.warning("tct evaluation doesn't support box_only, ignored")
    logger.info("performing tct evaluation, ignored iou_types.")
    return do_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )