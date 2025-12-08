import logging
import os

_logger = logging.getLogger('pixeltable')


def set_file_descriptor_limit(preferred_limit: int) -> None:
    """Checks and possibly updates the open file descriptor limit for the process.

    Note that there may be an OS-enforced upper bound on this limit, so this function may not always succeed in setting
    the preferred limit. It will log a warning and return normally in that case.
    """
    if os.name == 'nt':
        _logger.info('Skipping FD limit adjustment for Windows')
        return
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    _logger.info(f'Current RLIMIT_NOFILE soft limit: {soft_limit}, hard limit: {hard_limit}')
    if soft_limit < preferred_limit and soft_limit < hard_limit:
        new_limit = min(hard_limit, preferred_limit)
        _logger.info(f'Setting RLIMIT_NOFILE soft limit to: {new_limit}')
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
        soft_limit = new_limit

    if soft_limit < preferred_limit:
        _logger.warning(
            f'RLIMIT_NOFILE soft limit is {soft_limit}, which is less than the preferred {preferred_limit}. '
            'You may experience suboptimal network performance.'
        )
