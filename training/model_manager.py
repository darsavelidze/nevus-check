"""
Model Version Manager
Управление версиями обученных моделей
"""

import os
import json
from datetime import datetime
from pathlib import Path


class ModelManager:
    """Менеджер версий моделей"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.versions_dir = self.models_dir / 'versions'
        self.plots_dir = self.models_dir / 'plots'
        
        # Создаём папки если их нет
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def list_versions(self, sort_by='timestamp'):
        """Список всех версий моделей
        
        Args:
            sort_by: 'timestamp', 'accuracy', 'f1'
        """
        versions = []
        
        for metadata_file in self.versions_dir.glob('*_metadata.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                metadata['file'] = metadata_file.stem.replace('_metadata', '')
                versions.append(metadata)
        
        # Сортировка
        if sort_by == 'timestamp':
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
        elif sort_by == 'accuracy':
            versions.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
        elif sort_by == 'f1':
            versions.sort(key=lambda x: x['metrics']['f1'], reverse=True)
        
        return versions
    
    def get_best_model(self, metric='accuracy'):
        """Получить лучшую модель по метрике
        
        Args:
            metric: 'accuracy', 'f1', 'dice', 'precision'
        """
        versions = self.list_versions()
        
        if not versions:
            return None
        
        best = max(versions, key=lambda x: x['metrics'][metric])
        return best
    
    def get_version_info(self, version_name):
        """Получить информацию о конкретной версии"""
        metadata_path = self.versions_dir / f"{version_name}_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def set_as_current(self, version_name):
        """Установить версию как текущую (для использования в приложении)"""
        import shutil
        
        version_weights = self.versions_dir / f"{version_name}.h5"
        current_weights = self.models_dir / "skin_lesion_cnn_paper_final_weights.h5"
        
        if not version_weights.exists():
            raise FileNotFoundError(f"Weights not found: {version_weights}")
        
        # Копируем веса
        shutil.copy2(version_weights, current_weights)
        
        # Сохраняем информацию о текущей версии
        version_info = self.get_version_info(version_name)
        current_info_path = self.models_dir / "current_model_info.json"
        
        with open(current_info_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"✓ Version {version_name} set as current model")
        return True
    
    def compare_versions(self, version1, version2):
        """Сравнить две версии"""
        v1 = self.get_version_info(version1)
        v2 = self.get_version_info(version2)
        
        if not v1 or not v2:
            return None
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics_diff': {}
        }
        
        for metric in ['accuracy', 'f1', 'dice', 'precision', 'sensitivity', 'specificity']:
            diff = v2['metrics'][metric] - v1['metrics'][metric]
            comparison['metrics_diff'][metric] = {
                'v1': v1['metrics'][metric],
                'v2': v2['metrics'][metric],
                'diff': diff,
                'improvement': diff > 0
            }
        
        return comparison
    
    def print_versions_table(self):
        """Вывести таблицу всех версий"""
        versions = self.list_versions(sort_by='timestamp')
        
        if not versions:
            print("No model versions found.")
            return
        
        print("\n" + "="*100)
        print(f"{'Version':<40} {'Timestamp':<20} {'Accuracy':<10} {'F1':<10} {'Dice':<10}")
        print("="*100)
        
        for v in versions:
            print(f"{v['version']:<40} {v['timestamp']:<20} "
                  f"{v['metrics']['accuracy']*100:>6.2f}%   "
                  f"{v['metrics']['f1']*100:>6.2f}%   "
                  f"{v['metrics']['dice']:>6.4f}")
        
        print("="*100)
        
        # Показать лучшую модель
        best = self.get_best_model('accuracy')
        print(f"\n🏆 Best model (accuracy): {best['version']}")
        print(f"   Accuracy: {best['metrics']['accuracy']*100:.2f}%")
        print(f"   F1: {best['metrics']['f1']*100:.2f}%")


def main():
    """CLI для управления моделями"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Version Manager')
    parser.add_argument('command', choices=['list', 'best', 'set-current', 'info', 'compare'],
                      help='Command to execute')
    parser.add_argument('--version', help='Version name')
    parser.add_argument('--version2', help='Second version for comparison')
    parser.add_argument('--metric', default='accuracy', 
                      choices=['accuracy', 'f1', 'dice', 'precision'],
                      help='Metric for best model selection')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.command == 'list':
        manager.print_versions_table()
    
    elif args.command == 'best':
        best = manager.get_best_model(args.metric)
        if best:
            print(f"\nBest model ({args.metric}): {best['version']}")
            print(f"Metrics: {json.dumps(best['metrics'], indent=2)}")
        else:
            print("No models found")
    
    elif args.command == 'set-current':
        if not args.version:
            print("Error: --version required")
            return
        manager.set_as_current(args.version)
    
    elif args.command == 'info':
        if not args.version:
            print("Error: --version required")
            return
        info = manager.get_version_info(args.version)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Version {args.version} not found")
    
    elif args.command == 'compare':
        if not args.version or not args.version2:
            print("Error: --version and --version2 required")
            return
        comparison = manager.compare_versions(args.version, args.version2)
        if comparison:
            print(json.dumps(comparison, indent=2))
        else:
            print("Error: One or both versions not found")


if __name__ == '__main__':
    main()
