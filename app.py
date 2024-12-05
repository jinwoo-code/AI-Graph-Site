# app.py
import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용

from flask import Flask, render_template, request, send_file, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import os
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 세션을 위한 시크릿 키 설정 - 실제 운영시에는 보안성 높은 키를 사용해야 합니다

# 업로드된 파일과 모델을 저장할 디렉토리 생성
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
def create_figure():
    """matplotlib figure를 base64 인코딩된 이미지로 변환"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_visualizations(df):
    """데이터 시각화 생성"""
    visualizations = {}
    
    # 히스토그램
    plt.figure(figsize=(12, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols):
        plt.subplot(2, len(numeric_cols)//2 + 1, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    visualizations['histograms'] = create_figure()
    plt.close()
    
    # 상관관계 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    visualizations['correlation_heatmap'] = create_figure()
    plt.close()
    
    # 박스 플롯
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_cols):
        plt.subplot(2, len(numeric_cols)//2 + 1, i+1)
        sns.boxplot(x=df[col])
        plt.title(f'{col} Box Plot')
    plt.tight_layout()
    visualizations['boxplots'] = create_figure()
    plt.close()
    
    # 산점도 행렬
    plt.figure(figsize=(10, 10))
    sns.pairplot(df[numeric_cols])
    visualizations['pairplot'] = create_figure()
    plt.close()

    return visualizations

def generate_model_plots(model, X, y, y_pred, y_test, model_type):
    """모델 관련 시각화 생성"""
    plots = {}
    
    if model_type == 'regression':
        # 잔차 플롯
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, y_pred - y_test)  # y_test 사용
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plots['residual_plot'] = create_figure()
        plt.close()
        
    elif model_type == 'classification':
        # ROC 커브 등 추가 가능
        pass
        
    elif model_type == 'clustering':
        # 군집화 관련 시각화
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis')
        plt.title('Cluster Distribution')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plots['cluster_plot'] = create_figure()
        plt.close()
    
    # 특성 중요도 (회귀/분류)
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_)
        importances.sort_values(ascending=True).plot(kind='barh')
        plt.title('Feature Importance')
        plots['feature_importance'] = create_figure()
        plt.close()
    
    return plots



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 파일 업로드 처리 로직 추가
        return upload_file()  # 업로드 파일 처리 함수 호출
    return render_template('upload.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    if 'current_file' not in session:
        return jsonify({'error': '데이터 파일을 먼저 업로드해주세요.'}), 400

    # 데이터 로드
    df = pd.read_csv(session['current_file'])
    visualizations = {}

    # 사용자가 선택한 시각화 유형 가져오기
    selected_visualizations = request.form.getlist('visualization_types')

    if 'histogram' in selected_visualizations:
        plt.figure(figsize=(12, 6))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, len(numeric_cols)//2 + 1, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['histograms'] = create_figure()
        plt.close()

    if 'correlation_heatmap' in selected_visualizations:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        visualizations['correlation_heatmap'] = create_figure()
        plt.close()

    if 'boxplot' in selected_visualizations:
        plt.figure(figsize=(12, 6))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, len(numeric_cols)//2 + 1, i+1)
            sns.boxplot(x=df[col])
            plt.title(f'{col} Box Plot')
        plt.tight_layout()
        visualizations['boxplots'] = create_figure()
        plt.close()

    if 'pairplot' in selected_visualizations:
        plt.figure(figsize=(10, 10))
        sns.pairplot(df[numeric_cols])
        visualizations['pairplot'] = create_figure()
        plt.close()

    return render_template('visualization_results.html', visualizations=visualizations, columns=df.columns.tolist())



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': '파일이 제공되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다.'}), 400
        
        # 파일 저장
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 데이터 로드 및 분석
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({'error': f'파일 읽기 오류: {str(e)}'}), 400
        
        # 기본 통계량 및 정보 계산
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_count': df.isnull().sum().sum(),
            'missing_ratio': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(exclude=[np.number]).columns),
            'stats_table': df.describe().to_html(classes='table'),
        }
        
        # 시각화 생성
        visualizations = generate_visualizations(df)
        
        # 세션에 파일 경로 저장
        session['current_file'] = filepath
        
        # 분석 결과 페이지로 리디렉션
        return render_template('analysis.html',
                             filename=filename,
                             stats=stats,
                             visualizations=visualizations,
                             columns=df.columns.tolist())
    
    # GET 요청일 경우 업로드 페이지를 렌더링
    return render_template('upload.html')



@app.route('/train', methods=['POST'])
def train_model():
    if 'current_file' not in session:
        return jsonify({'error': '데이터 파일을 먼저 업로드해주세요.'}), 400
    
    # 파라미터 가져오기
    target = request.form.get('target')
    model_type = request.form.get('model_type')
    
    # 데이터 로드
    df = pd.read_csv(session['current_file'])
    
    # 특성과 타겟 분리
    X = df.drop(target, axis=1)
    y = df[target]
    
    # 전처리
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 모델 선택 및 학습
    metrics = {}
    
    if model_type == 'regression':
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 성능 지표 계산
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['cv_score'] = np.mean(cross_val_score(model, X_scaled, y, cv=5))
        metrics['score'] = metrics['r2']  # R² 점수 추가
        
    elif model_type == 'classification':
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 성능 지표 계산
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred)
        metrics['cv_score'] = np.mean(cross_val_score(model, X_scaled, y, cv=5))
        metrics['score'] = metrics['cv_score']  # 교차 검증 점수 추가
        
    elif model_type == 'clustering':
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X_scaled)
        y_pred = model.labels_
        
        # 성능 지표 계산
        metrics['silhouette_score'] = silhouette_score(X_scaled, y_pred)
    
    # 모델 저장
    model_filename = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
    joblib.dump(model, model_path)
    
    # 시각화 생성
    plots = generate_model_plots(model, X_scaled, y, y_pred, y_test, model_type)
    
    # 세션에 모델 경로 저장
    session['current_model'] = model_path
    
    return render_template('results.html',
                         model_type=model_type,
                         metrics=metrics,
                         plots=plots)


@app.route('/download_results', methods=['GET'])
def download_results():
    if 'current_file' not in session:
        return jsonify({'error': '분석 결과가 없습니다.'}), 400
    
    return send_file(session['current_file'], as_attachment=True)

@app.route('/download_model', methods=['GET'])
def download_model():
    if 'current_model' not in session:
        return jsonify({'error': '먼저 모델을 학습시켜주세요.'}), 400
    
    model_path = session['current_model']
    return send_file(model_path, as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'current_model' not in session:
        return jsonify({'error': '먼저 모델을 학습시켜주세요.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': '파일이 제공되지 않았습니다.'}), 400
    
    file = request.files['file']

    # 데이터 로드 및 전처리
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'파일 읽기 오류: {str(e)}'}), 400

    # 데이터 전처리
    model = joblib.load(session['current_model'])
    predictions = model.predict(df)

    # 결과를 DataFrame으로 변환
    results = pd.DataFrame({
        'prediction': predictions
    })

    # 결과 파일 저장
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    results.to_csv(result_path, index=False)
    
    return send_file(result_path, as_attachment=True)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'  # 세션을 위한 시크릿 키
    app.run(debug=True)