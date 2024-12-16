import yaml
from abc_smc_priors2 import run_calibration
from datetime import datetime, timedelta


def load_dates_config(config_file):
    """加载日期配置"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_abc(basin: str, model_id: str):
    """运行ABC校准"""
    # 加载配置
    config = load_dates_config('./dates_config.yaml')
    basin_config = config[basin]

    # 转换日期格式
    dates_dict = {
        'start_date': datetime.strptime(basin_config['start_date'], '%Y-%m-%d') - timedelta(days=30),
        'end_date': datetime.strptime(basin_config['end_date'], '%Y-%m-%d'),
        'vaccine_date': datetime.strptime(basin_config['vaccine_date'], '%Y-%m-%d'),
    }
    if basin!='London':
        dates_dict['t_alpha_org'] = datetime.strptime(basin_config['t_alpha_org'], '%Y-%m-%d') - timedelta(days=42)

    # 运行校准
    run_calibration(
        basin=basin,
        model_id=model_id,
        dates_dict=dates_dict,
        start_yw=basin_config['start_yw'],
        end_yw=basin_config['end_yw']
    )