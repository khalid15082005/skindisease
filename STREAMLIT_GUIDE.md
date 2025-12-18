# 🌐 Skin Disease AI - Streamlit Web Application Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements
```

### 2. Run the Web Application
```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## 📋 Features

### ✨ **User Interface**
- **Modern Design**: Clean, professional interface with custom styling
- **Responsive Layout**: Works on desktop and mobile devices
- **Image Upload**: Drag-and-drop or click to upload skin lesion images
- **Real-time Results**: Instant predictions with confidence scores
- **Visual Feedback**: Progress bars and color-coded results

### 🔬 **AI Capabilities**
- **6 Disease Classes**: Acne, Carcinoma, Eczema, Keratosis, Millia, Rosacea
- **High Accuracy**: 92% model accuracy on test data
- **Confidence Scoring**: Detailed confidence percentages for each prediction
- **Image Preprocessing**: Automatic resizing and normalization for optimal results

### 📊 **Results Display**
- **Top Prediction**: Most likely diagnosis with confidence score
- **All Predictions**: Complete breakdown of all 6 disease probabilities
- **Visual Bars**: Color-coded confidence bars for easy interpretation
- **Medical Disclaimer**: Important safety information for users

## 🎯 How to Use

1. **Upload Image**: Click "Choose a skin lesion image" or drag and drop
2. **Supported Formats**: PNG, JPG, JPEG
3. **Click Diagnose**: Press the "🔍 Diagnose" button
4. **View Results**: See predictions with confidence scores
5. **Interpret Results**: Use the confidence levels to understand reliability

## ⚠️ Important Notes

### **Medical Disclaimer**
- This tool is for **educational purposes only**
- Always consult a qualified dermatologist for medical diagnosis
- Not a substitute for professional medical advice

### **Model Requirements**
- Ensure your trained model is in the `model/` directory
- Model should be in TensorFlow SavedModel format
- Required files: `saved_model.pb`, `keras_metadata.pb`, `fingerprint.pb`

## 🔧 Troubleshooting

### **Common Issues**

1. **Model Not Found Error**
   ```
   Solution: Ensure model files are in the correct directory
   Check: model/saved_model.pb exists
   ```

2. **Import Errors**
   ```
   Solution: Install all dependencies
   Run: pip install -r requirements
   ```

3. **Image Upload Issues**
   ```
   Solution: Use supported formats (PNG, JPG, JPEG)
   Check: Image file is not corrupted
   ```

4. **Memory Issues**
   ```
   Solution: Use smaller images or restart the application
   Tip: Images are automatically resized to 299x299 pixels
   ```

## 📱 Mobile Compatibility

The web application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- Any device with a web browser

## 🎨 Customization

### **Modifying the Interface**
Edit `app.py` to customize:
- Colors and styling (CSS section)
- Layout and components
- Text and descriptions
- Model prediction logic

### **Adding New Features**
Consider adding:
- Batch image processing
- Export results to PDF
- User authentication
- Prediction history
- Advanced image analysis tools

## 📈 Performance Tips

1. **First Load**: May take a few seconds to load the model
2. **Subsequent Predictions**: Much faster due to model caching
3. **Image Size**: Larger images are automatically resized for optimal performance
4. **Memory Usage**: Monitor system resources for large-scale usage

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run app.py
```

### **Streamlit Cloud**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### **Docker Deployment**
Create a Dockerfile for containerized deployment

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure model files are in the correct location
4. Check the console for error messages

---

**🔬 Skin Disease AI | Built with Streamlit & TensorFlow**
