from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify # Đảm bảo có jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from slugify import slugify # Đảm bảo đã cài đặt: pip install python-slugify

app = Flask(__name__)

# --- Định nghĩa đường dẫn ---
MODEL_PATH = 'model/best_model_finetune.keras'
LABELS_PATH = 'labels.txt'
PRODUCT_DATA_PATH = 'product_data.json' # Đường dẫn file thông tin chi tiết giống mèo
PRODUCTS_CATALOG_PATH = 'products_catalog.json' # Đường dẫn file catalog sản phẩm
UPLOAD_FOLDER = os.path.join('static', 'uploads') # Thư mục lưu ảnh tải lên
EVAL_DIR = os.path.join('static', 'evaluation_output') # Thư mục lưu báo cáo đánh giá
MISCLASSIFIED_DIR = os.path.join(EVAL_DIR, 'misclassified') # Thư mục lưu ảnh dự đoán sai

# --- Tải Model và Nhãn (Labels) ---
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")

class_names = []
try:
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Labels loaded successfully.")
except FileNotFoundError:
    print(f"Error: labels.txt not found at {LABELS_PATH}")
except Exception as e:
    print(f"Error reading labels from {LABELS_PATH}: {e}")

# --- Tải Mô tả giống mèo ---
cat_breed_info = {}
try:
    with open(PRODUCT_DATA_PATH, 'r', encoding='utf-8') as f:
        cat_breed_info = json.load(f)
    print("Cat breed info loaded successfully.")
except FileNotFoundError:
    print(f"Error: product_data.json not found at {PRODUCT_DATA_PATH}. Descriptions will not be available.")
except json.JSONDecodeError as e:
    print(f"Error decoding product_data.json: {e}. Check JSON format.")
except Exception as e:
    print(f"Error loading cat breed info: {e}")

# --- Tải Danh mục sản phẩm ---
products_catalog = {}
try:
    with open(PRODUCTS_CATALOG_PATH, 'r', encoding='utf-8') as f:
        products_catalog = json.load(f)
    print("Products catalog loaded successfully.")
except FileNotFoundError:
    print(f"Error: products_catalog.json not found at {PRODUCTS_CATALOG_PATH}. Product listings will be empty.")
except json.JSONDecodeError as e:
    print(f"Error decoding products_catalog.json: {e}. Check JSON format.")
except Exception as e:
    print(f"Error loading products catalog: {e}")

# --- Ánh xạ Giống mèo sang Sản phẩm gợi ý CỤ THỂ (từ catalog) ---
BREED_SPECIFIC_RECOMMENDATIONS = {
    "Abyssinian": [
        {"category": "food", "name": "Thức ăn hạt cao cấp cho mèo mọi lứa tuổi"},
        {"category": "toy", "name": "Đồ chơi banh len đan thủ công"},
        {"category": "accessory", "name": "Cây cào móng dạng trụ với đồ chơi"}
    ],
    "Siamese": [
        {"category": "food", "name": "Pate thức ăn ướt vị cá ngừ Fancy Feast"},
        {"category": "toy", "name": "Đồ chơi laser cho mèo tự động"},
        {"category": "accessory", "name": "Vòng cổ mèo da mềm có chuông"}
    ],
    "Persian": [
        {"category": "food", "name": "Thức ăn khô Royal Canin Kitten"},
        {"category": "toy", "name": "Chuột đồ chơi vải lông có chuông"},
        {"category": "accessory", "name": "Lược chải lông mèo chuyên dụng"}
    ],
    "Maine Coon": [
        {"category": "food", "name": "Thức ăn hạt cao cấp cho mèo mọi lứa tuổi"},
        {"category": "toy", "name": "Tháp bóng 3 tầng tương tác"},
        {"category": "accessory", "name": "Đệm ngủ tròn mềm mại và ấm áp"}
    ],
    "Bengal": [
        {"category": "food", "name": "Snack thưởng cá hồi sấy khô Mera"},
        {"category": "toy", "name": "Cần câu mèo gắn lông vũ tự nhiên"},
        {"category": "accessory", "name": "Cây cào móng dạng trụ với đồ chơi"}
    ],
    "Default": [
        {"category": "food", "name": "Thức ăn hạt Whiskas hương vị cá thu"},
        {"category": "toy", "name": "Bộ 5 quả bóng lông mèo mềm mại"},
        {"category": "accessory", "name": "Bát ăn đôi chống trượt bằng thép không gỉ"}
    ]
}

def get_recommended_products_by_breed(breed_label, catalog):
    recommendations_config = BREED_SPECIFIC_RECOMMENDATIONS.get(breed_label, BREED_SPECIFIC_RECOMMENDATIONS['Default'])
    recommended_products = []
    for rec in recommendations_config:
        category_name = rec["category"]
        product_name = rec["name"]
        if category_name in catalog:
            for i, p in enumerate(catalog[category_name]):
                if p["name"] == product_name:
                    product_meta = p.copy() # Tạo bản sao để thêm metadata
                    product_meta['category_slug'] = category_name
                    product_meta['index'] = i
                    recommended_products.append(product_meta)
                    break # Tìm thấy sản phẩm này thì chuyển sang sản phẩm gợi ý tiếp theo
    return recommended_products

# --- Hàm sắp xếp sản phẩm ---
def sort_products(products_list, sort_by):
    if not products_list:
        return []

    if sort_by == 'price_asc':
        return sorted(products_list, key=lambda p: p.get('price', 0))
    elif sort_by == 'price_desc':
        return sorted(products_list, key=lambda p: p.get('price', 0), reverse=True)
    elif sort_by == 'name_asc':
        return sorted(products_list, key=lambda p: p.get('name', '').lower())
    elif sort_by == 'name_desc':
        return sorted(products_list, key=lambda p: p.get('name', '').lower(), reverse=True)
    elif sort_by == 'popular' or sort_by == 'newest':
        return products_list 
    else:
        return products_list 

# --- Tạo các thư mục cần thiết ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# --- Các Route của Ứng dụng (Render HTML) ---

@app.route('/')
def index():
    featured_products = []
    if "food" in products_catalog:
        for i, p in enumerate(products_catalog["food"][:2]):
            product_with_meta = p.copy()
            product_with_meta['category_slug'] = 'food'
            product_with_meta['index'] = i
            featured_products.append(product_with_meta)
    if "toy" in products_catalog:
        for i, p in enumerate(products_catalog["toy"][:1]):
            product_with_meta = p.copy()
            product_with_meta['category_slug'] = 'toy'
            product_with_meta['index'] = i
            featured_products.append(product_with_meta)
    if "accessory" in products_catalog:
        for i, p in enumerate(products_catalog["accessory"][:1]):
            product_with_meta = p.copy()
            product_with_meta['category_slug'] = 'accessory'
            product_with_meta['index'] = i
            featured_products.append(product_with_meta)
    
    return render_template('index.html', featured_products=featured_products)

@app.route('/food')
def food():
    sort_by = request.args.get('sort_by', 'popular')
    food_items = products_catalog.get("food", [])
    total_food_items = len(food_items)
    sorted_food_items = sort_products(food_items, sort_by)
    return render_template('food.html', products=sorted_food_items, 
                           category_name='food', current_sort=sort_by, total_products=total_food_items)

@app.route('/toy')
def toy():
    sort_by = request.args.get('sort_by', 'popular')
    toy_items = products_catalog.get("toy", [])
    total_toy_items = len(toy_items)
    sorted_toy_items = sort_products(toy_items, sort_by)
    return render_template('toy.html', products=sorted_toy_items, 
                           category_name='toy', current_sort=sort_by, total_products=total_toy_items)

@app.route('/accessory')
def accessory():
    sort_by = request.args.get('sort_by', 'popular')
    accessory_items = products_catalog.get("accessory", [])
    total_accessory_items = len(accessory_items)
    sorted_accessory_items = sort_products(accessory_items, sort_by)
    return render_template('accessory.html', products=sorted_accessory_items, 
                           category_name='accessory', current_sort=sort_by, total_products=total_accessory_items)

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/advise')
def advise():
    return render_template('advise.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(url_for('advise'))

    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return redirect(url_for('advise'))

    if not model:
        print("Prediction attempted but model is not loaded.")
        return "Error: Model not loaded. Please check server logs.", 500

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)
        print(f"File saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {file.filename}: {e}")
        return "Error: Could not save the uploaded file.", 500

    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        pred = model.predict(img_array, verbose=0)
        probabilities = pred[0]

        pred_idx = np.argmax(probabilities)
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown Breed"

        top_indices = np.argsort(probabilities)[::-1]
        top_predictions_list = []
        num_explicit_top = 3

        for i in range(min(num_explicit_top, len(class_names))):
            idx = top_indices[i]
            if idx < len(class_names):
                label = class_names[idx]
                prob = probabilities[idx]
                top_predictions_list.append(f"{label} ({prob*100:.2f}%)")

        if len(class_names) > num_explicit_top:
            remaining_probability = sum(probabilities[idx] for idx in top_indices[num_explicit_top:]) * 100
            if remaining_probability > 1:
                top_predictions_list.append(f"Khác ({remaining_probability:.2f}%)")
            elif remaining_probability > 0:
                top_predictions_list.append(f"Khác (Dưới 1%)")

        # Lấy thông tin chi tiết giống mèo
        breed_info = cat_breed_info.get(pred_label, None)
        if breed_info is None:
            breed_info = {
                "description": "Không có thông tin chi tiết cho giống mèo này.",
                "origin": "Không rõ",
                "personality": "Không rõ",
                "care": "Không rõ",
                "appearance": "Không rõ",
                "health": "Không rõ",
                "fun_fact": "Không có thông tin."
            }
        
        # LẤY CÁC SẢN PHẨM GỢI Ý THỰC TẾ TỪ CATALOG DỰA TRÊN GIỐNG MÈO
        recommended_products = get_recommended_products_by_breed(pred_label, products_catalog)

        return render_template('result.html',
                               filename=file.filename,
                               predicted_label=pred_label,
                               top_predictions_list=top_predictions_list,
                               breed_info=breed_info,
                               recommended_products=recommended_products)
    except Exception as e:
        print(f"Error during image processing or prediction for {file.filename}: {e}")
        return "Error: An issue occurred during image processing or prediction.", 500

@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/evaluation')
def evaluation():
    report_path = os.path.join(EVAL_DIR, 'classification_report.txt')
    report_text = "Không tìm thấy báo cáo phân loại (classification_report.txt)."
    accuracy = "N/A"
    num_test_images = "N/A"
    confusion_matrix_img_path = 'evaluation_output/confusion_matrix.png'

    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
                for line in report_text.splitlines():
                    if "accuracy" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                accuracy_val = float(parts[-2])
                                num_test_images_val = int(parts[-1])
                                accuracy = f"{accuracy_val * 100:.2f}"
                                num_test_images = str(num_test_images_val)
                            except ValueError:
                                print(f"Could not parse accuracy/support from line: {e}")
                                pass
                        break
        except Exception as e:
            report_text = f"Lỗi khi đọc báo cáo: {e}"
            print(f"Error reading classification_report.txt: {e}")
    else:
        print(f"classification_report.txt not found at {report_path}")

    misclassified_images = []
    if os.path.exists(MISCLASSIFIED_DIR):
        for img_name in os.listdir(MISCLASSIFIED_DIR):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                misclassified_images.append(f"evaluation_output/misclassified/{img_name}")
        misclassified_images.sort()

    if not os.path.exists(os.path.join('static', confusion_matrix_img_path)):
        print(f"Warning: Confusion matrix image not found at {os.path.join('static', confusion_matrix_img_path)}")

    return render_template('evaluation.html',
                           report_text=report_text,
                           confusion_matrix_img=confusion_matrix_img_path,
                           misclassified_images=misclassified_images,
                           num_test_images=num_test_images,
                           accuracy=accuracy)

@app.route('/download-report')
def download_report():
    return send_from_directory(EVAL_DIR, 'classification_report.txt', as_attachment=True)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    sort_by = request.args.get('sort_by', 'popular')
    found_products = []
    if query:
        lower_query = query.lower()
        for category, products in products_catalog.items():
            for i, product in enumerate(products):
                if lower_query in product.get("name", "").lower() or \
                   (product.get("description") and lower_query in product["description"].lower()):
                    
                    product_with_meta = product.copy()
                    product_with_meta['category_slug'] = category
                    product_with_meta['index'] = i
                    found_products.append(product_with_meta)
    
    total_found_products = len(found_products)
    found_products = sort_products(found_products, sort_by) # Sắp xếp kết quả tìm kiếm
    
    return render_template('search_results.html', query=query, products=found_products, 
                           total_results=total_found_products, current_sort=sort_by)

@app.route('/product/<category_name>/<int:product_index>')
def product_detail(category_name, product_index):
    category_products = products_catalog.get(category_name, [])
    
    if 0 <= product_index < len(category_products):
        product = category_products[product_index]
        product_with_meta = product.copy()
        product_with_meta['category_slug'] = category_name
        product_with_meta['index'] = product_index
        return render_template('product_detail.html', product=product_with_meta)
    else:
        return render_template('404.html'), 404


# --- CÁC ROUTES API MỚI CHO CHỨC NĂNG SẮP XẾP AJAX ---

@app.route('/api/products/<category_slug>')
def api_get_products_by_category(category_slug):
    sort_by = request.args.get('sort_by', 'popular')
    category_products = products_catalog.get(category_slug, [])
    
    # Thêm metadata (category_slug, index) cho từng sản phẩm trước khi sắp xếp và gửi đi
    products_with_meta = []
    for i, p in enumerate(category_products):
        p_copy = p.copy()
        p_copy['category_slug'] = category_slug
        p_copy['index'] = i
        products_with_meta.append(p_copy)

    sorted_products = sort_products(products_with_meta, sort_by)
    
    return jsonify({
        'products': sorted_products,
        'total_products': len(category_products) # Tổng số sản phẩm trong danh mục này
    })

@app.route('/api/search_results') # Đổi tên thành search_results để rõ ràng là API cho trang search_results
def api_search_products():
    query = request.args.get('q', '').strip()
    sort_by = request.args.get('sort_by', 'popular')
    found_products = []
    
    if query:
        lower_query = query.lower()
        for category, products in products_catalog.items():
            for i, product in enumerate(products):
                if lower_query in product.get("name", "").lower() or \
                   (product.get("description") and lower_query in product["description"].lower()):
                    
                    product_with_meta = product.copy()
                    product_with_meta['category_slug'] = category
                    product_with_meta['index'] = i
                    found_products.append(product_with_meta)
    
    total_found_products = len(found_products) # Tổng số sản phẩm tìm thấy trước khi sắp xếp
    found_products = sort_products(found_products, sort_by) # Sắp xếp kết quả tìm kiếm

    return jsonify({
        'products': found_products,
        'total_products': total_found_products, # Tổng số kết quả tìm thấy
        'query': query
    })


# --- Xử lý lỗi tùy chỉnh ---
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


#if __name__ == '__main__':
 #   app.run(debug=True)
if __name__ == '__main__':
    # Sử dụng Waitress cho môi trường phát triển cục bộ trên Windows
    from waitress import serve
    print("WARNING: Using Waitress for local development. For production deployment, consider using Gunicorn on Linux servers or Waitress.")
    # Waitress sẽ phục vụ ứng dụng Flask (đối tượng 'app') trên host và port đã chỉ định
    serve(app, host='0.0.0.0', port=5000)