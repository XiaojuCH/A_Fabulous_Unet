import csv

input_csv = "./results/master_evaluation_full.csv"
output_csv = "./results/master_evaluation_fixed.csv"

def fix_csv():
    print("🛠️ 正在进行 CSV 外科手术，剔除旧版脏数据...")
    
    with open(input_csv, 'r', encoding='utf-8') as fin, open(output_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # 写入完美的 8 列新表头
        writer.writerow(['Fold', 'Image_ID', 'Modality', 'Model', 'Prompt', 'Dice', 'HD95', 'ASD'])
        
        valid_count = 0
        for row in reader:
            # 只保留长度严格为 8 的新流水线数据，并过滤掉可能重复写入的表头
            if len(row) == 8 and row[0] != 'Fold':
                writer.writerow(row)
                valid_count += 1
                
    print(f"✅ 手术成功！共抢救回 {valid_count} 行有效预测数据。")
    print(f"📁 干净的数据已保存至: {output_csv}")

if __name__ == "__main__":
    fix_csv()