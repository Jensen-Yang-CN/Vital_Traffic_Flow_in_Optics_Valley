import requests
import time
import csv
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

# ==================== 配置区域 ====================
# ⚠️ 安全约定：API Key 一律从环境变量读取，禁止硬编码进源码。
# 请先在「高德开放平台 → 应用管理」申请 Web 服务类型的 Key，然后设置环境变量：
#   PowerShell :  $env:AMAP_API_KEY="你的Key"      # 仅当前窗口有效
#                 或 [Environment]::SetEnvironmentVariable("AMAP_API_KEY","你的Key","User")
#   CMD        :  set AMAP_API_KEY=你的Key
#   Bash / zsh :  export AMAP_API_KEY="你的Key"
#   也可写入项目根目录的 .env 文件（.env 已在 .gitignore 中，不会被提交）
API_KEY = os.environ.get("AMAP_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit(
        "❌ 未检测到环境变量 AMAP_API_KEY，采集脚本无法启动。\n"
        "   请先到高德开放平台申请 Web 服务 API Key，然后设置环境变量，例如：\n"
        '     PowerShell : $env:AMAP_API_KEY="你的Key"\n'
        '     CMD        : set AMAP_API_KEY=你的Key\n'
        '     Bash       : export AMAP_API_KEY="你的Key"\n'
        "   设置完成后重新运行本脚本。"
    )

# RECTANGLE = "114.396,30.503;114.402,30.509"
RECTANGLE = "114.397,30.504;114.401,30.508"
ROAD_LEVEL = "6"
INTERVAL_SECONDS = 120
CSV_FILENAME = "guanggu_traffic_0430.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STATUS_MAP = {'0': '未知', '1': '畅通', '2': '缓行', '3': '拥堵'}


# ==================== 限流控制 ====================

class RateLimiter:
    def __init__(self, max_calls_per_second: int = 2):
        self.max_calls_per_second = max_calls_per_second
        self.last_request_time = 0
        self.min_interval = 1.0 / max_calls_per_second

    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        self.last_request_time = time.time()


rate_limiter = RateLimiter(max_calls_per_second=2)


# ==================== API 请求 ====================

def fetch_traffic_data(rectangle: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """获取矩形区域交通路况数据"""
    rate_limiter.wait_if_needed()

    url = "https://restapi.amap.com/v3/traffic/status/rectangle"
    params = {
        'rectangle': rectangle,
        'key': API_KEY,
        'level': ROAD_LEVEL,
        'extensions': 'all'
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"正在请求 API (尝试 {attempt + 1}/{max_retries})...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == '1':
                logger.info("API 请求成功")
                return data
            else:
                info = data.get('info', '未知错误')
                logger.warning(f"API 返回错误: {info}")
                return None

        except Exception as e:
            logger.warning(f"请求异常: {e}")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    return None


# ==================== 数据解析 ====================

def parse_traffic_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """解析交通路况数据"""
    result = {
        'status': None,
        'status_desc': '未知',
        'speed': None,
        'expedite': None,
        'congested': None,
        'blocked': None,
        'unknown': None,
        'description': '',
        'road_count': 0,
        'roads': []
    }

    try:
        traffic_info = data.get('trafficinfo', {})

        # 解析整体评估
        evaluation = traffic_info.get('evaluation', {})
        result['status'] = evaluation.get('status')
        result['speed'] = evaluation.get('speed')
        result['expedite'] = evaluation.get('expedite')
        result['congested'] = evaluation.get('congested')
        result['blocked'] = evaluation.get('blocked')
        result['unknown'] = evaluation.get('unknown')
        result['description'] = evaluation.get('description', '')
        result['status_desc'] = STATUS_MAP.get(str(result['status']), '未知')

        # 解析道路列表
        roads = traffic_info.get('roads', [])
        result['road_count'] = len(roads)

        for road in roads:
            road_status = road.get('status')
            road_info = {
                'name': road.get('name', ''),
                'status': road_status,
                'status_text': STATUS_MAP.get(str(road_status), '未知'),
                'speed': road.get('speed'),
                'direction': road.get('direction', ''),  # 方向信息：从XXX到XXX
                'angle': road.get('angle', ''),
                'lcodes': road.get('lcodes', '')
            }
            result['roads'].append(road_info)

        logger.info(f"成功解析 {len(roads)} 条道路")

    except Exception as e:
        logger.error(f"解析数据失败: {e}")

    return result


# ==================== CSV 文件操作 ====================

#

def init_csv_file(filename: str):
    """初始化 CSV 文件 - 仅在文件不存在时创建表头"""
    if not os.path.exists(filename):
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '采集时间',
                '整体路况代码',
                '整体路况描述',
                '平均车速(km/h)',
                '畅通率(%)',
                '拥堵率(%)',
                '阻塞率(%)',
                '道路名称',
                '道路状态',
                '速度(km/h)',
                '行驶方向',
                '方向角度'
            ])
        logger.info(f"已创建 CSV 文件: {filename}")
    else:
        logger.info(f"CSV 文件已存在，将追加数据: {filename}")


# def save_to_csv(filename: str, data: Dict[str, Any]):
#     """
#     保存数据到 CSV - 每条道路单独一行
#     """
#     try:
#         collect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#         with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
#             writer = csv.writer(f)
#
#             # 如果没有道路数据，只写入整体信息
#             if not data['roads']:
#                 writer.writerow([
#                     collect_time,
#                     data.get('status', ''),
#                     data.get('status_desc', ''),
#                     data.get('speed', ''),
#                     data.get('expedite', ''),
#                     data.get('congested', ''),
#                     data.get('blocked', ''),
#                     '', '', '', '', ''
#                 ])
#             else:
#                 # 每条道路单独一行
#                 for road in data['roads']:
#                     writer.writerow([
#                         collect_time,
#                         data.get('status', ''),
#                         data.get('status_desc', ''),
#                         data.get('speed', ''),
#                         data.get('expedite', ''),
#                         data.get('congested', ''),
#                         data.get('blocked', ''),
#                         road.get('name', ''),
#                         road.get('status_text', ''),
#                         road.get('speed', ''),
#                         road.get('direction', ''),
#                         road.get('angle', '')
#                     ])
#
#         logger.info(f"数据已保存: {len(data['roads'])} 条道路记录")
#
#     except Exception as e:
#         logger.error(f"保存 CSV 失败: {e}")
def save_to_csv(filename: str, data: Dict[str, Any]):
    """保存数据到 CSV - 每条道路单独一行"""
    try:
        collect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 手动计算整体平均速度（从各道路速度取平均）
        speeds = []
        for road in data['roads']:
            if road.get('speed') and road['speed'].isdigit():
                speeds.append(int(road['speed']))

        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else None

        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            for road in data['roads']:
                writer.writerow([
                    collect_time,
                    data.get('status', ''),
                    data.get('status_desc', ''),
                    avg_speed,  # 使用计算出的平均速度
                    data.get('expedite', ''),
                    data.get('congested', ''),
                    data.get('blocked', ''),
                    road.get('name', ''),
                    road.get('status_text', ''),
                    road.get('speed', ''),
                    road.get('direction', ''),
                    road.get('angle', '')
                ])

        logger.info(f"数据已保存: 平均速度={avg_speed}km/h, {len(data['roads'])} 条道路")

    except Exception as e:
        logger.error(f"保存 CSV 失败: {e}")


def print_traffic_info(data: Dict[str, Any]):
    """打印路况信息到控制台"""
    print(f"\n{'=' * 70}")
    print(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    print(f"整体路况: {data['status_desc']}")
    print(f"平均车速: {data['speed']} km/h" if data['speed'] else "平均车速: 暂无数据")
    print(f"畅通率: {data['expedite']}")
    print(f"拥堵率: {data['congested']}")
    print(f"阻塞率: {data['blocked']}")
    print(f"描述: {data['description']}")
    print(f"道路总数: {data['road_count']}")

    if data['roads']:
        print(f"\n 道路详情:")
        for road in data['roads']:
            speed_info = f"{road['speed']}km/h" if road['speed'] else "速度未知"
            direction_info = f" [{road['direction']}]" if road['direction'] else ""
            print(f"   • {road['name']}: {road['status_text']}, {speed_info}{direction_info}")


def fetch_and_save():
    """采集并保存数据"""
    logger.info("=" * 50)
    logger.info("开始采集光谷交通路况数据")

    raw_data = fetch_traffic_data(RECTANGLE)
    if not raw_data:
        logger.warning("获取数据失败，本次采集跳过")
        return

    parsed_data = parse_traffic_info(raw_data)
    print_traffic_info(parsed_data)
    save_to_csv(CSV_FILENAME, parsed_data)
    logger.info("采集完成")


#

def test_single_fetch():
    """单次测试采集"""
    print("\n" + "=" * 70)
    print("单次测试采集模式")
    print("=" * 70)
    print(f"矩形范围: {RECTANGLE}")

    raw_data = fetch_traffic_data(RECTANGLE)

    if raw_data:
        parsed_data = parse_traffic_info(raw_data)

        print(f"\n整体统计:")
        print(f"  整体路况: {parsed_data['status_desc']}")
        print(f"  平均车速: {parsed_data['speed']} km/h")
        print(f"  道路总数: {parsed_data['road_count']}")

        if parsed_data['roads']:
            print(f"\n道路详情:")
            for road in parsed_data['roads']:
                speed_info = f"{road['speed']}km/h" if road['speed'] else "速度未知"
                direction_info = f" [{road['direction']}]" if road['direction'] else ""
                print(f"  • {road['name']}: {road['status_text']}, {speed_info}{direction_info}")

        # 修改：只在文件不存在时才创建，否则追加
        if not os.path.exists(CSV_FILENAME):
            init_csv_file(CSV_FILENAME)

        save_to_csv(CSV_FILENAME, parsed_data)
        print(f"测试数据已保存到 {CSV_FILENAME}")
        # 注意：这里不再重新初始化文件，避免覆盖
    else:
        logger.error("获取数据失败")
        return None


def run_scheduler():
    """运行定时采集任务"""
    print("\n" + "=" * 60)
    print(" 定时采集任务已启动")
    print("=" * 60)
    print(f"采集间隔: {INTERVAL_SECONDS} 秒 ({INTERVAL_SECONDS / 60:.1f} 分钟)")
    print(f"目标区域: {RECTANGLE}")
    print(f"数据保存文件: {CSV_FILENAME}")
    print("=" * 60)

    start_time = datetime.now()
    logger.info(f"任务开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        while True:
            try:
                fetch_and_save()
            except Exception as e:
                logger.error(f"采集过程中发生异常: {e}")

            logger.info(f"等待 {INTERVAL_SECONDS} 秒后继续采集...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("用户中断任务")
    finally:
        end_time = datetime.now()
        logger.info(f"任务结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


def test_api_connection():
    """测试 API 连接"""
    logger.info("测试 API 连接...")
    data = fetch_traffic_data(RECTANGLE, max_retries=1)
    if data:
        parsed = parse_traffic_info(data)
        logger.info(f"API 连接测试成功！")
        logger.info(f"当前路况: {parsed['status_desc']}, 道路数: {parsed['road_count']}")
        return True
    return False


# ==================== 主程序 ====================

#
def main():
    print("\n" + "=" * 60)
    print("🚦 高德交通态势采集工具 - 光谷转盘版")
    print("=" * 60)

    print(f"\n配置信息:")
    print(f"  矩形范围: {RECTANGLE}")
    print(f"  采集间隔: {INTERVAL_SECONDS} 秒")
    print(f"  保存文件: {CSV_FILENAME}")

    # 显示已有数据条数
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            data_count = len(lines) - 1  # 减去表头
            print(f"  已有数据: {data_count} 条道路记录")

    print("\n🔌 正在测试 API 连接...")
    if not test_api_connection():
        print("\nAPI 连接测试失败！")
        print("请检查 API Key 是否正确")
        return

    print("\nAPI 连接测试成功！")

    print("\n是否进行单次测试采集？(y/n)")
    if input().lower() == 'y':
        test_single_fetch()

    print("\n是否启动定时采集任务？(y/n)")
    print("提示: 输入 y 后，脚本将每 2 分钟自动采集一次数据")

    if input().lower() == 'y':
        # 确保文件存在（但不覆盖）
        if not os.path.exists(CSV_FILENAME):
            init_csv_file(CSV_FILENAME)
        run_scheduler()
    else:
        print("已退出")


if __name__ == "__main__":
    main()