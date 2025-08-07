document.addEventListener('DOMContentLoaded', function() {
    // Función de depuración
    function debugObject(obj, label = "Objeto") {
        console.log(`------- ${label} -------`);
        if (!obj) {
            console.log("VALOR NULO O UNDEFINED");
            return;
        }
        
        if (typeof obj === 'object') {
            console.log("Tipo:", typeof obj);
            console.log("Propiedades:", Object.keys(obj));
            for (const key of Object.keys(obj)) {
                const value = obj[key];
                const type = typeof value;
                const preview = type === 'object' ? (Array.isArray(value) ? `Array(${value.length})` : `Object`) : String(value).substring(0, 50);
                console.log(`  - ${key}: [${type}] ${preview}`);
            }
        } else {
            console.log("Valor:", obj);
        }
        console.log("------------------------");
    }
    
    // Referencias a elementos del DOM globales
    const uploadForm = document.getElementById('upload-form');
    const forecastForm = document.getElementById('forecast-form');
    const articleSelect = document.getElementById('article-select');
    const forecastingChart = document.getElementById('forecasting-chart');
    const chartLoader = document.getElementById('chart-loader');
    const forecastTableBody = document.getElementById('forecast-table-body');
    const optimizeButton = document.getElementById('optimize-button');
    
    // Referencias adicionales a elementos del DOM
    const monthsInput = document.getElementById('months-input');
    const forecastMonths = document.getElementById('forecast-months');
    
    // Actualizar el contador de meses pronosticados cuando cambia el valor
    if (monthsInput && forecastMonths) {
        monthsInput.addEventListener('change', function() {
            forecastMonths.textContent = this.value;
        });
        
        // Establecer valor inicial
        forecastMonths.textContent = monthsInput.value;
    }
    
    // Chart.js - Inicializar gráfico vacío
    let chart = null;
    
    // Cargar artículos si hay datos cargados
    fetchArticles();
    
    // Event Listeners
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    if (forecastForm) {
        forecastForm.addEventListener('submit', handleForecastGeneration);
    }
    
    if (optimizeButton) {
        optimizeButton.addEventListener('click', handleModelOptimization);
    }
    
    // Mostrar nombre del archivo cuando se selecciona y cargar automáticamente
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileInfo = document.getElementById('file-info');
            const fileName = document.getElementById('file-name');
            const uploadButton = document.getElementById('upload-button');
            const uploadSpinner = document.getElementById('upload-spinner');
            
            if (this.files.length > 0) {
                if (fileInfo && fileName) {
                    fileInfo.style.display = 'block';
                    fileName.textContent = this.files[0].name;
                }
                
                // Cambiar botón a estado de carga
                if (uploadButton && uploadSpinner) {
                    uploadButton.disabled = true;
                    uploadButton.classList.add('opacity-75');
                    uploadButton.querySelector('.button-text').textContent = 'Cargando...';
                    uploadSpinner.classList.remove('hidden');
                }
                
                // Ejecutar formulario automáticamente
                setTimeout(() => {
                    if (uploadForm) {
                        uploadForm.dispatchEvent(new Event('submit'));
                    }
                }, 300);
            } else {
                if (fileInfo) {
                    fileInfo.style.display = 'none';
                }
            }
        });
    }
    
    // Clear file selection
    const clearFileButton = document.getElementById('clear-file');
    if (clearFileButton) {
        clearFileButton.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent the click from triggering the file input
            
            const fileInput = document.getElementById('file-input');
            const fileInfo = document.getElementById('file-info');
            
            if (fileInput) {
                fileInput.value = ''; // Clear the file input
            }
            
            if (fileInfo) {
                fileInfo.style.display = 'none';
            }
        });
    }
    
    // Manejo de carga de archivos
    function handleFileUpload(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];
        const uploadButton = document.getElementById('upload-button');
        const uploadSpinner = document.getElementById('upload-spinner');
        
        if (!file) {
            alert('Por favor seleccione un archivo');
            resetUploadButton();
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Mostrar info del archivo seleccionado
        const fileInfo = document.getElementById('file-info');
        const fileName = document.getElementById('file-name');
        
        if (fileInfo && fileName) {
            fileInfo.style.display = 'block';
            fileName.textContent = file.name;
        }
        
        fetch('/api/load-data', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resetUploadButton();
            
            if (data.success) {
                // Mostrar notificación más elegante que un alert
                showNotification('Datos cargados correctamente', 'success');
                updateStats(data.stats);
                populateArticleSelect(data.articles);
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resetUploadButton();
            showNotification('Error al cargar el archivo', 'error');
            return; // Añadido para detener la ejecución
        });
    }
    
    // Función para resetear el botón de carga
    function resetUploadButton() {
        const uploadButton = document.getElementById('upload-button');
        const uploadSpinner = document.getElementById('upload-spinner');
        
        if (uploadButton && uploadSpinner) {
            uploadButton.disabled = false;
            uploadButton.classList.remove('opacity-75');
            uploadButton.querySelector('.button-text').textContent = 'Cargar Datos';
            uploadSpinner.classList.add('hidden');
        }
    }
    
    // Función para mostrar notificaciones
    function showNotification(message, type = 'info') {
        const notificationContainer = document.getElementById('notification-container');
        
        if (!notificationContainer) {
            // Crear contenedor de notificaciones si no existe
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'fixed top-4 right-4 z-50';
            document.body.appendChild(container);
        }
        
        // Crear notificación
        const notification = document.createElement('div');
        notification.className = `mb-3 p-3 rounded-lg shadow-md flex items-center transition-opacity duration-500 transform translate-x-0`;
        
        // Estilo según tipo
        if (type === 'success') {
            notification.classList.add('bg-green-50', 'border-l-4', 'border-green-500', 'text-green-700');
            notification.innerHTML = `<i class="fas fa-check-circle mr-2 text-green-500"></i> ${message}`;
        } else if (type === 'error') {
            notification.classList.add('bg-red-50', 'border-l-4', 'border-red-500', 'text-red-700');
            notification.innerHTML = `<i class="fas fa-exclamation-circle mr-2 text-red-500"></i> ${message}`;
        } else {
            notification.classList.add('bg-blue-50', 'border-l-4', 'border-blue-500', 'text-blue-700');
            notification.innerHTML = `<i class="fas fa-info-circle mr-2 text-blue-500"></i> ${message}`;
        }
        
        // Añadir a contenedor
        document.getElementById('notification-container').appendChild(notification);
        
        // Eliminar después de 3 segundos
        setTimeout(() => {
            notification.classList.add('opacity-0');
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 3000);
    }
    
    // Obtener artículos disponibles
    function fetchArticles() {
        fetch('/api/get-articles')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                populateArticleSelect(data.articles);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            return; // Añadido para detener la ejecución
        });
    }
    
    // Poblar selector de artículos
    function populateArticleSelect(articles) {
        if (!articleSelect) return;
        
        // Limpiar selector
        articleSelect.innerHTML = '<option value="">Seleccione un artículo...</option>';
        
        // Añadir opciones
        articles.forEach(article => {
            const option = document.createElement('option');
            option.value = article.id;
            option.textContent = article.name;
            articleSelect.appendChild(option);
        });

        // Actualizar también el selector de stock de seguridad
        const ssArticleSelect = document.getElementById('ss-article-select');
        if (ssArticleSelect) {
            // Limpiar selector
            ssArticleSelect.innerHTML = '<option value="">Seleccione un artículo...</option>';
            
            // Añadir opciones
            articles.forEach(article => {
                const option = document.createElement('option');
                option.value = article.id;
                option.textContent = article.name;
                ssArticleSelect.appendChild(option);
            });
        }
    }
    
    // Actualizar estadísticas en el panel
    function updateStats(stats) {
        const totalArticles = document.getElementById('total-articles');
        const totalRecords = document.getElementById('total-records');
        
        if (totalArticles) totalArticles.textContent = stats.total_articles;
        if (totalRecords) totalRecords.textContent = stats.total_records;
    }
    
    // Generar pronósticos
    function handleForecastGeneration(e, useOptimized = false) {
        if (e) {
            e.preventDefault();
        }
        
        const articleId = articleSelect.value;
        const target = document.getElementById('target-select').value;
        const months = document.getElementById('months-input').value;
        
        if (!articleId) {
            alert('Por favor seleccione un artículo');
            return;
        }
        
        // Obtener modelos seleccionados
        const selectedModels = [];
        if (document.getElementById('sarima-check') && document.getElementById('sarima-check').checked) {
            selectedModels.push('SARIMA');
        }
        if (document.getElementById('lstm-check') && document.getElementById('lstm-check').checked) {
            selectedModels.push('LSTM');
        }
        if (document.getElementById('gru-check') && document.getElementById('gru-check').checked) {
            selectedModels.push('GRU');
        }
        if (document.getElementById('xgboost-check') && document.getElementById('xgboost-check').checked) {
            selectedModels.push('XGBOOST');
        }
        
        // Verificar que al menos un modelo esté seleccionado
        if (selectedModels.length === 0) {
            alert('Por favor seleccione al menos un modelo de pronóstico');
            return;
        }
        
        // Mostrar loader
        if (chartLoader) {
            chartLoader.classList.remove('hidden');
            chartLoader.classList.add('flex');
        }
        
        // Preparar parámetros
        const requestParams = {
            article_id: articleId,
            target: target,
            steps: parseInt(months),
            models: selectedModels  // Añadir modelos seleccionados a la solicitud
        };
        
        // Si se solicita específicamente el modelo optimizado
        if (useOptimized) {
            requestParams.use_optimized = true;
        }
        
        console.log("Enviando solicitud de pronóstico para:", requestParams);
        
        // Enviar solicitud para generar pronósticos
        fetch('/api/generate-forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestParams)
        })
        .then(response => response.json())
        .then(data => {
            if (chartLoader) {
                chartLoader.classList.add('hidden');
                chartLoader.classList.remove('flex');
            }
            
            console.log("Respuesta de pronóstico recibida:", data);
            
            if (data.success) {
                // Actualizar UI con resultados
                updateChart(data.data, selectedModels);  // Pasar los modelos seleccionados
                updateMetrics(data.data);
                updateForecastTable(data.data);
                
                // Establecer mejor modelo
                const bestModel = data.data.best_model;
                const bestModelBadge = document.getElementById('best-model-badge');
                const bestModelName = document.getElementById('best-model-name');
                
                if (bestModelBadge) {
                    bestModelBadge.className = `model-badge model-${bestModel.toLowerCase()}`;
                    bestModelBadge.textContent = bestModel;
                }
                
                if (bestModelName) {
                    bestModelName.textContent = getBestModelName(bestModel);
                }
                
                // Configurar enlaces de exportación
                setupExportLinks(articleId, target);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            if (chartLoader) {
                chartLoader.classList.add('hidden');
                chartLoader.classList.remove('flex');
            }
            console.error('Error:', error);
            alert('Error al generar pronósticos');
            return; // Añadido para detener la ejecución
        });
    }
    
    // Optimización de cualquier modelo (reemplaza a handleXGBoostOptimization)
    function handleModelOptimization() {
        const articleId = articleSelect.value;
        const target = document.getElementById('target-select').value;
        const modelType = document.getElementById('optimizer-model-select').value;
        
        if (!articleId) {
            alert('Por favor seleccione un artículo para optimizar');
            return;
        }
        
        // Configurar modal
        const totalTrials = 50;
        document.getElementById('total-trials').textContent = totalTrials;
        document.getElementById('trials-completed').textContent = '0';
        document.getElementById('optimization-progress').style.width = '0%';
        
        // Actualizar el nombre del modelo en el modal
        const modelSelect = document.getElementById('optimizer-model-select');
        document.getElementById('optimization-model-name').textContent = modelSelect.options[modelSelect.selectedIndex].text;
        
        // Mostrar modal usando Tailwind's hidden/flex classes
        const optimizationModal = document.getElementById('optimization-modal');
        optimizationModal.classList.remove('hidden');
        optimizationModal.classList.add('flex');
        
        console.log(`Iniciando optimización de ${modelType} para artículo ID: ${articleId}, target: ${target}`);
        
        // Iniciar optimización
        fetch('/api/optimize-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                article_id: articleId,
                model_type: modelType,
                target: target,
                n_trials: totalTrials
            })
        })
        .then(response => {
            console.log("Respuesta recibida:", response.status);
            return response.json();
        })
        .then(data => {
            console.log("Datos de respuesta:", data);
            
            if (data.success) {
                const taskId = data.task_id;
                console.log("Tarea de optimización iniciada con ID:", taskId);
                
                // Variables para detectar bloqueos
                let noProgressCount = 0;
                let lastTrialCount = 0;
                
                // Iniciar polling para verificar estado
                const statusCheckInterval = setInterval(() => {
                    console.log(`Verificando estado de tarea ${taskId}...`);
                    
                    fetch(`/api/optimize-xgboost-status/${taskId}`)
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`Error en respuesta: ${response.status}`);
                            }
                            return response.json();
                        })
                        .then(statusData => {
                            console.log("Estado de la tarea:", statusData);
                            
                            if (statusData.success) {
                                const task = statusData.task;
                                const progressBar = document.getElementById('optimization-progress');
                                const trialsText = document.getElementById('trials-completed');
                                
                                // Actualizar progreso
                                progressBar.style.width = `${task.progress}%`;
                                
                                // Actualizar número de trials completados
                                if (task.result && task.result.current_trial !== undefined) {
                                    trialsText.textContent = task.result.current_trial;
                                    
                                    // Verificar si hay progreso
                                    if (task.result.current_trial === lastTrialCount) {
                                        noProgressCount++;
                                        console.log(`Sin progreso por ${noProgressCount} intervalos`);
                                    } else {
                                        noProgressCount = 0;
                                        lastTrialCount = task.result.current_trial;
                                        console.log(`Progreso detectado: ${task.result.current_trial}/${task.result.total_trials}`);
                                    }
                                    
                                    // Si no hay progreso por mucho tiempo, mostrar mensaje
                                    if (noProgressCount > 15) {
                                        console.warn("No se detecta progreso en la optimización");
                                        progressBar.classList.add('bg-warning');
                                        
                                        // Si estamos realmente bloqueados (30+ verificaciones sin progreso)
                                        if (noProgressCount > 30) {
                                            clearInterval(statusCheckInterval);
                                            optimizationModal.classList.add('hidden');
                                            optimizationModal.classList.remove('flex');
                                            alert('La optimización parece estar bloqueada. Por favor, inténtelo de nuevo.');
                                        }
                                    }
                                }
                                
                                // Verificar si la tarea ha finalizado
                                if (task.status === 'completed') {
                                    clearInterval(statusCheckInterval);
                                    console.log("¡Optimización completada!");
                                    
                                    // Marcar como completado
                                    progressBar.style.width = '100%';
                                    trialsText.textContent = totalTrials;
                                    
                                    // Esperar un momento para mostrar el progreso completo
                                    setTimeout(() => {
                                        // Cerrar modal
                                        optimizationModal.classList.add('hidden');
                                        optimizationModal.classList.remove('flex');
                                        
                                        // Mostrar resultados
                                        if (task.result && task.result.error) {
                                            alert(`La optimización completó con errores: ${task.result.error}`);
                                        } else {
                                            alert(`Optimización de ${modelType} completada exitosamente. Se encontraron los mejores parámetros.`);
                                            
                                            // Generar pronósticos utilizando el modelo optimizado
                                            handleForecastGeneration(null, true);
                                        }
                                    }, 1000);
                                } else if (task.status === 'failed') {
                                    clearInterval(statusCheckInterval);
                                    console.error("Optimización fallida:", task.error);
                                    optimizationModal.classList.add('hidden');
                                    optimizationModal.classList.remove('flex');
                                    alert(`Error en la optimización: ${task.error || 'Error desconocido'}`);
                                }
                            } else {
                                console.error("Error al verificar estado:", statusData.error);
                                noProgressCount++;
                                
                                if (noProgressCount > 10) {
                                    clearInterval(statusCheckInterval);
                                    optimizationModal.classList.add('hidden');
                                    optimizationModal.classList.remove('flex');
                                    alert(`Error al verificar estado: ${statusData.error || 'Error desconocido'}`);
                                }
                            }
                        })
                        .catch(error => {
                            console.error("Error en petición de estado:", error);
                            noProgressCount++;
                            
                            if (noProgressCount > 5) {
                                clearInterval(statusCheckInterval);
                                optimizationModal.classList.add('hidden');
                                optimizationModal.classList.remove('flex');
                                alert(`Error al verificar estado: ${error.message}`);
                                return;
                            }
                        });
                }, 2000); // Consultar cada 2 segundos
                
                // Establecer un timeout global para evitar bloqueos indefinidos
                setTimeout(() => {
                    if (statusCheckInterval) {
                        clearInterval(statusCheckInterval);
                        optimizationModal.classList.add('hidden');
                        optimizationModal.classList.remove('flex');
                        alert('La optimización ha excedido el tiempo límite. Por favor, inténtelo de nuevo con menos pruebas.');
                    }
                }, 300000); // 5 minutos máximo
                
            } else {
                console.error("Error al iniciar optimización:", data.error);
                optimizationModal.classList.add('hidden');
                optimizationModal.classList.remove('flex');
                alert('Error al iniciar la optimización: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error en solicitud:', error);
            optimizationModal.classList.add('hidden');
            optimizationModal.classList.remove('flex');
            alert('Error al comunicarse con el servidor: ' + error.message);
            return; // Añadido para detener la ejecución
        });
    }
    
    // Actualizar gráfico con datos de pronóstico
    function updateChart(data, selectedModels) {
        console.log("Iniciando actualización del gráfico con modelos seleccionados:", selectedModels);
        const canvas = document.getElementById('forecast-chart');
        
        if (!canvas) {
            console.error("No se encontró el elemento canvas 'forecast-chart'");
            return;
        }
        
        // Asegurarse de que el canvas esté visible y tenga dimensiones
        const container = canvas.parentElement;
        if (container) {
            container.style.minHeight = '300px';
        }
        
        // Obtener datos históricos
        fetch(`/api/article-data/${articleSelect.value}`)
        .then(response => response.json())
        .then(histData => {
            if (!histData.success) {
                console.error("Error obteniendo datos históricos:", histData.error);
                return;
            }
            
            console.log("Datos históricos recibidos:", histData.data);
            
            const target = document.getElementById('target-select').value;
            const article = data.article;
            
            // Preparar datos históricos
            const historicalDates = histData.data.dates;
            // Intentar diferentes opciones para el nombre de la propiedad
            let historicalValues = histData.data.cantidades || 
                                  histData.data.producción || 
                                  [];
            
            console.log("Fechas históricas:", historicalDates);
            console.log("Valores históricos:", historicalValues);
            
            if (!historicalValues || !Array.isArray(historicalValues)) {
                console.error("No se pudieron obtener valores históricos válidos");
                console.log("Datos disponibles:", Object.keys(histData.data));
                return;
            }
            
            // Prepara etiquetas y datos
            const labels = [];
            const datasets = [];
            
            // Convierte fechas a etiquetas para el eje X
            for (let i = 0; i < historicalDates.length; i++) {
                labels.push(formatDateShort(historicalDates[i]));
            }
            
            // Datos históricos
            const historicalDataset = {
                label: 'Datos Históricos',
                data: historicalValues,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                borderWidth: 2
            };
            
            // Preparar datasets para pronósticos
            const forecastDatasets = [];
            const colors = {
                'SARIMA': '#e74c3c',
                'LSTM': '#2ecc71',
                'GRU': '#9b59b6',
                'XGBOOST': '#f39c12'
            };
            
            // Añadir las etiquetas de fechas futuras
            const allForecastDates = new Set();
            
            // Obtener todas las fechas futuras de los modelos seleccionados
            for (const [model, forecast] of Object.entries(data.forecasts)) {
                // Solo procesar si el modelo está seleccionado
                if (selectedModels.includes(model) && forecast && forecast.dates && Array.isArray(forecast.dates)) {
                    forecast.dates.forEach(date => {
                        allForecastDates.add(formatDateShort(date));
                    });
                }
            }
            
            // Ordenar fechas futuras y añadirlas a las etiquetas
            const sortedFutureDates = Array.from(allForecastDates).sort();
            labels.push(...sortedFutureDates);
            
            // Añadir dataset histórico
            datasets.push(historicalDataset);
            
            // Extender el dataset histórico con nulls para las fechas futuras
            historicalDataset.data = [...historicalDataset.data, ...Array(sortedFutureDates.length).fill(null)];
            
            // Crear datasets solo para los modelos seleccionados
            for (const [model, forecast] of Object.entries(data.forecasts)) {
                // Solo procesar si el modelo está seleccionado
                if (!selectedModels.includes(model) || !forecast || !forecast.dates || !forecast.values) continue;
                
                // Crear array con nulls para fechas históricas
                const forecastData = Array(historicalDates.length + sortedFutureDates.length).fill(null);
                
                // Mapear datos de pronóstico a las etiquetas correspondientes
                for (let i = 0; i < forecast.dates.length; i++) {
                    const dateLabel = formatDateShort(forecast.dates[i]);
                    const labelIndex = labels.indexOf(dateLabel);
                    if (labelIndex !== -1) {
                        forecastData[labelIndex] = forecast.values[i];
                    }
                }
                
                // Crear dataset para este modelo
                forecastDatasets.push({
                    label: `Pronóstico ${model}`,
                    data: forecastData,
                    borderColor: colors[model] || '#e74c3c',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    pointRadius: 3,
                    tension: 0.4,
                    borderWidth: 2
                });
                
                // Añadir al array principal de datasets
                datasets.push(forecastDatasets[forecastDatasets.length - 1]);
            }
            
            // Configurar gráfico
            const config = {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Fecha'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: target
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `Pronóstico de ${target} - ${article}`,
                            font: {
                                size: 16
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        },
                        legend: {
                            position: 'top',
                        }
                    }
                }
            };
            
            // Destruir gráfico anterior si existe
            if (chart instanceof Chart) {
                console.log("Destruyendo gráfico anterior");
                chart.destroy();
            }
            
            // Crear nuevo gráfico
            try {
                console.log("Creando nuevo gráfico");
                const ctx = canvas.getContext('2d');
                chart = new Chart(ctx, config);
                console.log("Gráfico creado exitosamente");
            } catch (error) {
                console.error("Error al crear el gráfico:", error);
            }
        })
        .catch(error => {
            console.error('Error al obtener datos históricos:', error);
        });
    }
    
    // Actualizar métricas en UI
    function updateMetrics(data) {
        if (!data.best_model || !data.forecasts[data.best_model] || !data.forecasts[data.best_model].metrics) {
            console.warn("No hay métricas disponibles para mostrar");
            return;
        }
        
        const metrics = data.forecasts[data.best_model].metrics;
        
        const metricMSE = document.getElementById('metric-mse');
        const metricRMSE = document.getElementById('metric-rmse');
        const metricMAE = document.getElementById('metric-mae');
        const metricMAPE = document.getElementById('metric-mape');
        const metricR2 = document.getElementById('metric-r2');
        
        if (metricMSE) metricMSE.textContent = metrics.MSE.toFixed(4);
        if (metricRMSE) metricRMSE.textContent = metrics.RMSE.toFixed(4);
        if (metricMAE) metricMAE.textContent = metrics.MAE.toFixed(4);
        if (metricMAPE) metricMAPE.textContent = metrics.MAPE.toFixed(2) + '%';
        if (metricR2) metricR2.textContent = metrics.R2.toFixed(4);
    }
    
    // Actualizar tabla de pronósticos
    function updateForecastTable(data) {
        if (!forecastTableBody) {
            console.warn("No se encuentra el elemento forecastTableBody");
            return;
        }
        
        forecastTableBody.innerHTML = '';
        
        if (!data.best_model || !data.forecasts[data.best_model]) {
            console.warn("No hay datos de pronóstico para mostrar en la tabla");
            return;
        }
        
        const bestForecast = data.forecasts[data.best_model];
        const target = document.getElementById('target-select').value;
        const modelName = data.best_model;
        
        if (!bestForecast.dates || !bestForecast.values) {
            console.warn("Datos de pronóstico incompletos");
            return;
        }
        
        for (let i = 0; i < bestForecast.dates.length; i++) {
            const row = document.createElement('tr');
            
            const dateCell = document.createElement('td');
            dateCell.textContent = formatDate(bestForecast.dates[i]);
            
            const articleCell = document.createElement('td');
            articleCell.textContent = data.article;
            
            const modelCell = document.createElement('td');
            const modelBadge = document.createElement('span');
            modelBadge.classList.add('model-badge');
            modelBadge.classList.add(`model-${modelName.toLowerCase()}`);
            modelBadge.textContent = modelName;
            modelCell.appendChild(modelBadge);
            
            const valueCell = document.createElement('td');
            valueCell.textContent = bestForecast.values[i].toFixed(3);
            
            row.appendChild(dateCell);
            row.appendChild(articleCell);
            row.appendChild(modelCell);
            row.appendChild(valueCell);
            
            forecastTableBody.appendChild(row);
        }
    }
    
    // Configurar enlaces de exportación
    function setupExportLinks(articleId, target) {
        const downloadForecast = document.getElementById('download-forecast');
        const downloadChart = document.getElementById('download-chart');
        
        if (downloadForecast) {
            downloadForecast.onclick = function() {
                window.location.href = `/api/export-forecast/${articleId}/${target}`;
            };
        }
        
        if (downloadChart) {
            downloadChart.onclick = function() {
                if (chart instanceof Chart) {
                    const link = document.createElement('a');
                    link.download = `pronostico-${articleId}-${target}.png`;
                    link.href = chart.toBase64Image();
                    link.click();
                } else {
                    console.warn("No hay gráfico disponible para descargar");
                    alert("El gráfico no está disponible para descargar");
                }
            };
        }
    }
    
    // Utilidades
    function getBestModelName(model) {
        const models = {
            'SARIMA': 'Modelo Estadístico SARIMA',
            'LSTM': 'Red Neuronal LSTM',
            'GRU': 'Red Neuronal GRU',
            'XGBOOST': 'Modelo XGBoost Regressor'
        };
        return models[model] || model;
    }
    
    function formatDate(dateStr) {
        try {
            const date = new Date(dateStr);
            const month = date.getMonth() + 1; // getMonth() devuelve 0-11
            const year = date.getFullYear();
            // Nombres de meses en español
            const monthNames = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
            return `${monthNames[month-1]} ${year}`;
        } catch (error) {
            console.warn("Error al formatear fecha:", error);
            return dateStr;
        }
    }
    
    // Función para formatear fechas en formato corto para etiquetas
    function formatDateShort(dateStr) {
        try {
            const date = new Date(dateStr);
            // Formato simple como "MM/YYYY"
            return `${date.getMonth() + 1}/${date.getFullYear()}`;
        } catch (error) {
            console.warn("Error al formatear fecha:", error);
            return dateStr;
        }
    }
    
    // Notificar carga completa
    console.log("Script app.js cargado completamente");

    // Referencias a elementos del DOM para Stock de Seguridad
    const leadtimeForm = document.getElementById('leadtime-form');
    const safetyStockForm = document.getElementById('safety-stock-form');
    const ssArticleSelect = document.getElementById('ss-article-select');
    const ssMethodSelect = document.getElementById('ss-method-select');
    const reviewPeriodContainer = document.getElementById('review-period-container');
    const calculateAllSSButton = document.getElementById('calculate-all-ss');
    const exportSSButton = document.getElementById('export-ss');
    const ssResultsContainer = document.getElementById('ss-results-container');
    const ssAllResultsContainer = document.getElementById('ss-all-results-container');
    
    // Inicialización
    initSafetyStockUI();
    
    // Event Listeners
    if (leadtimeForm) {
        leadtimeForm.addEventListener('submit', handleLeadTimeUpload);
    }
    
    if (safetyStockForm) {
        safetyStockForm.addEventListener('submit', handleSafetyStockCalculation);
    }
    
    if (ssMethodSelect) {
        ssMethodSelect.addEventListener('change', toggleReviewPeriodVisibility);
    }
    
    if (calculateAllSSButton) {
        calculateAllSSButton.addEventListener('click', handleCalculateAllSafetyStocks);
    }
    
    if (exportSSButton) {
        exportSSButton.addEventListener('click', handleExportSafetyStock);
    }
    
    // Función para inicializar la UI de Stock de Seguridad
    function initSafetyStockUI() {
        // Obtener el selector de artículos directamente
        const articleSelect = document.getElementById('article-select');
        const ssArticleSelect = document.getElementById('ss-article-select');
        
        // Sincronizar el selector de artículos con el selector principal
        if (ssArticleSelect && articleSelect) {
            // Clonar las opciones del selector principal
            Array.from(articleSelect.options).forEach(option => {
                if (option.value !== "") {  // Omitir la opción vacía inicial
                    const newOption = document.createElement('option');
                    newOption.value = option.value;
                    newOption.textContent = option.textContent;
                    ssArticleSelect.appendChild(newOption);
                }
            });
        }
        
        // Configurar visibilidad inicial del campo de período de revisión
        toggleReviewPeriodVisibility();
    }
    
    // Función para manejar la carga de tiempos de entrega
    function handleLeadTimeUpload(e) {
        e.preventDefault();
        console.log("Función handleLeadTimeUpload ejecutada");
        
        const fileInput = document.getElementById('leadtime-file');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            showNotification('Por favor seleccione un archivo Excel', 'error');
            console.error("No se seleccionó ningún archivo");
            return;
        }
        
        const file = fileInput.files[0];
        console.log("Enviando archivo:", file.name, "Tamaño:", file.size, "Tipo:", file.type);
        
        // Mostrar información del formulario
        const formData = new FormData();
        formData.append('file', file);
        
        // Reportar inicio
        showNotification('Enviando archivo...', 'info');
        
        // Deshabilitar botón durante el envío
        const submitButton = document.querySelector('#leadtime-form button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
        }
        
        // Especificar timeout más largo
        fetch('/api/load-leadtimes', {
            method: 'POST',
            body: formData,
            // Aumentar timeout
            timeout: 30000
        })
        .then(response => {
            console.log("Respuesta del servidor:", response.status);
            if (!response.ok) {
                throw new Error(`Error en la respuesta del servidor: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Datos recibidos:", data);
            if (data.success) {
                showNotification(`Tiempos de entrega cargados: ${data.count} artículos`, 'success');
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error en fetch:', error);
            showNotification('Error al cargar archivo: ' + error.message, 'error');
        })
        .finally(() => {
            // Re-habilitar botón
            if (submitButton) {
                submitButton.disabled = false;
            }
        });
    }
    
    // Función para manejar el cálculo de stock de seguridad
    function handleSafetyStockCalculation(e) {
        e.preventDefault();
        
        const articleId = ssArticleSelect.value;
        const target = document.getElementById('ss-target-select').value;
        const method = ssMethodSelect.value;
        const serviceLevel = document.getElementById('ss-service-level').value / 100;  // Convertir de porcentaje a decimal
        
        if (!articleId) {
            showNotification('Por favor seleccione un artículo', 'error');
            return;
        }
        
        // Obtener horizonte de pronóstico y si usar pronósticos
        const forecastHorizon = document.getElementById('ss-forecast-horizon').value;
        const useForecasts = document.getElementById('ss-use-forecasts').checked;
        
        // Parámetros adicionales según el método
        let params = `?target=${target}&method=${method}&service_level=${serviceLevel}&forecast_horizon=${forecastHorizon}&use_forecasts=${useForecasts}`;
        
        if (method === 'review') {
            const reviewPeriod = document.getElementById('ss-review-period').value;
            params += `&review_period=${reviewPeriod}`;
        }
        
        console.log(`Calculando stock de seguridad para artículo ID ${articleId} con parámetros: ${params}`);
        
        fetch(`/api/safety-stock/${articleId}${params}`)
        .then(response => {
            console.log("Respuesta status:", response.status);
            return response.json();
        })
        .then(data => {
            console.log("Datos recibidos:", data);
            if (data.success) {
                displaySafetyStockResults(data.data);
                // Mostrar contenedor de resultados
                ssResultsContainer.classList.remove('hidden');
                // Ocultar resultados de todos los artículos si están visibles
                ssAllResultsContainer.classList.add('hidden');
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error al calcular stock de seguridad:', error);
            showNotification('Error al calcular stock de seguridad: ' + error.message, 'error');
        });
    }
    
    // Función para manejar el cálculo de stock de seguridad para todos los artículos
    function handleCalculateAllSafetyStocks() {
        const target = document.getElementById('ss-target-select').value;
        const method = ssMethodSelect.value;
        const serviceLevel = document.getElementById('ss-service-level').value / 100;  // Convertir de porcentaje a decimal
        
        // Obtener horizonte de pronóstico y si usar pronósticos
        const forecastHorizon = document.getElementById('ss-forecast-horizon').value;
        const useForecasts = document.getElementById('ss-use-forecasts').checked;
        
        const params = `?target=${target}&method=${method}&service_level=${serviceLevel}&forecast_horizon=${forecastHorizon}&use_forecasts=${useForecasts}`;
        
        // Mostrar notificación de carga
        showNotification('Calculando stock de seguridad para todos los artículos...', 'info');
        
        fetch(`/api/safety-stock${params}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayAllSafetyStockResults(data.data);
                // Mostrar contenedor de resultados de todos los artículos
                ssAllResultsContainer.classList.remove('hidden');
                // Ocultar resultados individuales si están visibles
                ssResultsContainer.classList.add('hidden');
                
                // Mostrar notificación de éxito
                showNotification(`Stock de seguridad calculado para ${Object.keys(data.data).length} artículos`, 'success');
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error al calcular stock de seguridad para todos los artículos', 'error');
        });
    }
    
    // Función para manejar la exportación de resultados de stock de seguridad
    function handleExportSafetyStock() {
        const target = document.getElementById('ss-target-select').value;
        const method = ssMethodSelect.value;
        const serviceLevel = document.getElementById('ss-service-level').value / 100;  // Convertir de porcentaje a decimal
        
        fetch('/api/export-safety-stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target: target,
                method: method,
                service_level: serviceLevel
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Resultados exportados correctamente', 'success');
                // Redirigir a la URL de descarga
                if (data.download_url) {
                    window.location.href = data.download_url;
                }
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error al exportar resultados', 'error');
        });
    }
    
    // Función para mostrar/ocultar el campo de período de revisión
    function toggleReviewPeriodVisibility() {
        if (ssMethodSelect && reviewPeriodContainer) {
            if (ssMethodSelect.value === 'review') {
                reviewPeriodContainer.classList.remove('hidden');
            } else {
                reviewPeriodContainer.classList.add('hidden');
            }
        }
    }
    
    // Función para mostrar los resultados del cálculo de stock de seguridad
    function displaySafetyStockResults(data) {
        // Actualizar valores en la UI
        document.getElementById('ss-result-value').textContent = data.safety_stock.toFixed(2);
        document.getElementById('ss-leadtime-value').textContent = data.leadtime_days;
        document.getElementById('ss-service-level-value').textContent = (data.service_level * 100).toFixed(1) + '%';
        
        // Mostrar el punto de reorden
        if (data.reorder_point !== undefined) {
            document.getElementById('ss-reorder-point-value').textContent = data.reorder_point.toFixed(2);
        } else {
            document.getElementById('ss-reorder-point-value').textContent = 'N/A';
        }

        // Estadísticas de la demanda
        document.getElementById('ss-avg-demand').textContent = data.avg_demand.toFixed(2);
        document.getElementById('ss-std-dev').textContent = data.std_dev.toFixed(2);
        document.getElementById('ss-data-points').textContent = data.data_points;
        
        // Tipo de demanda si está disponible
        if (data.demand_analysis && data.demand_analysis.demand_type) {
            document.getElementById('ss-demand-type').textContent = data.demand_analysis.demand_type;
        } else {
            document.getElementById('ss-demand-type').textContent = 'No disponible';
        }
        
        // Parámetros de cálculo
        document.getElementById('ss-method-used').textContent = getMethodName(data.method);
        document.getElementById('ss-z-factor').textContent = getZFactorForServiceLevel(data.service_level).toFixed(2);
        document.getElementById('ss-leadtime-periods').textContent = data.leadtime_periods.toFixed(2);
        document.getElementById('ss-article-name').textContent = data.article;
        
        // Manejar stock de seguridad por mes si está disponible
        const monthlyContainer = document.getElementById('ss-monthly-container');
        const monthlyTable = document.getElementById('ss-monthly-table');
        
        if (data.safety_stocks_by_month && Object.keys(data.safety_stocks_by_month).length > 0) {
            // Mostrar contenedor mensual
            monthlyContainer.classList.remove('hidden');
            
            // Limpiar tabla
            monthlyTable.innerHTML = '';
            
            // Obtener pronósticos si están disponibles
            const forecasts = data.forecasts || [];
            
            // Índice para acceder a los pronósticos
            let forecastIndex = 0;
            
            // Añadir filas a la tabla
            for (const date in data.safety_stocks_by_month) {
                const safetyStock = data.safety_stocks_by_month[date];
                
                const row = document.createElement('tr');
                
                // Celda de fecha
                const dateCell = document.createElement('td');
                dateCell.className = 'px-3 py-2 whitespace-nowrap text-sm text-gray-900';
                dateCell.textContent = formatDate(date);
                
                // Celda de stock de seguridad
                const ssCell = document.createElement('td');
                ssCell.className = 'px-3 py-2 whitespace-nowrap text-sm text-gray-900 font-medium';
                ssCell.textContent = safetyStock.toFixed(2);
                
                // Celda de pronóstico
                const forecastCell = document.createElement('td');
                forecastCell.className = 'px-3 py-2 whitespace-nowrap text-sm text-gray-900';
                
                // Añadir pronóstico si está disponible
                if (forecasts.length > forecastIndex) {
                    forecastCell.textContent = forecasts[forecastIndex].toFixed(2);
                    forecastIndex++;
                } else {
                    forecastCell.textContent = 'N/A';
                }
                
                // Añadir celdas a la fila
                row.appendChild(dateCell);
                row.appendChild(ssCell);
                row.appendChild(forecastCell);
                
                // Añadir fila a la tabla
                monthlyTable.appendChild(row);
            }
        } else {
            // Ocultar contenedor mensual
            monthlyContainer.classList.add('hidden');
        }
        
        // Manejar información del modelo si está disponible
        const modelInfoContainer = document.getElementById('ss-model-info-container');
        
        if (data.best_model) {
            // Mostrar contenedor de información del modelo
            modelInfoContainer.classList.remove('hidden');
            
            // Actualizar información del modelo
            document.getElementById('ss-model-name').textContent = data.best_model.name;
            
            if (data.best_model.metrics) {
                const metrics = data.best_model.metrics;
                document.getElementById('ss-model-rmse').textContent = metrics.RMSE ? metrics.RMSE.toFixed(2) : 'N/A';
                document.getElementById('ss-model-mae').textContent = metrics.MAE ? metrics.MAE.toFixed(2) : 'N/A';
            }
        } else {
            // Ocultar contenedor de información del modelo
            modelInfoContainer.classList.add('hidden');
        }
    }
    
    // Función para formatear fecha
    function formatDate(dateStr) {
        try {
            const date = new Date(dateStr);
            const options = { year: 'numeric', month: 'short' };
            return date.toLocaleDateString('es-ES', options);
        } catch {
            return dateStr;
        }
    }
    
    // Función para mostrar los resultados del cálculo de stock de seguridad para todos los artículos
    function displayAllSafetyStockResults(allData) {
        const tableBody = document.getElementById('ss-all-results-body');
    
    if (!tableBody) return;
    
    // Limpiar tabla
    tableBody.innerHTML = '';
    
    // Añadir filas a la tabla con la nueva columna de punto de reorden
    for (const article in allData) {
        const data = allData[article];
        
        const row = document.createElement('tr');
        
        // Celda de artículo
        const articleCell = document.createElement('td');
        articleCell.className = 'px-6 py-4 whitespace-normal text-sm text-gray-900';
        articleCell.textContent = article;
        
        // Celda de stock de seguridad
        const ssCell = document.createElement('td');
        ssCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
        ssCell.textContent = data.safety_stock.toFixed(2);
        
        // Celda de punto de reorden (nueva)
        const ropCell = document.createElement('td');
        ropCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
        if (data.reorder_point !== undefined) {
            ropCell.textContent = data.reorder_point.toFixed(2);
        } else {
            ropCell.textContent = 'N/A';
        }

        // Celda de tiempo de entrega
        const ltCell = document.createElement('td');
        ltCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
        ltCell.textContent = data.leadtime_days + ' días';
            
            // Mostrar stock de seguridad con detalles mensuales si están disponibles
            if (data.safety_stocks_by_month && Object.keys(data.safety_stocks_by_month).length > 0) {
                const ssValue = document.createElement('div');
                ssValue.className = 'font-medium';
                ssValue.textContent = data.safety_stock.toFixed(2);
                
                // Crear un botón para mostrar/ocultar detalles
                const detailsBtn = document.createElement('button');
                detailsBtn.className = 'text-xs text-indigo-600 hover:text-indigo-900 mt-1';
                detailsBtn.textContent = '+ Ver por mes';
                detailsBtn.type = 'button';
                
                // Crear el contenedor de detalles (inicialmente oculto)
                const detailsContainer = document.createElement('div');
                detailsContainer.className = 'hidden mt-2 text-xs space-y-1 bg-gray-50 p-2 rounded';
                
                // Añadir cada valor mensual al contenedor de detalles
                for (const [date, value] of Object.entries(data.safety_stocks_by_month)) {
                    const detail = document.createElement('div');
                    detail.className = 'flex justify-between';
                    
                    const dateSpan = document.createElement('span');
                    dateSpan.className = 'text-gray-600';
                    dateSpan.textContent = formatDate(date) + ':';
                    
                    const valueSpan = document.createElement('span');
                    valueSpan.className = 'font-medium';
                    valueSpan.textContent = value.toFixed(2);
                    
                    detail.appendChild(dateSpan);
                    detail.appendChild(valueSpan);
                    detailsContainer.appendChild(detail);
                }
                
                // Añadir evento para mostrar/ocultar detalles
                detailsBtn.addEventListener('click', function() {
                    if (detailsContainer.classList.contains('hidden')) {
                        detailsContainer.classList.remove('hidden');
                        detailsBtn.textContent = '- Ocultar detalles';
                    } else {
                        detailsContainer.classList.add('hidden');
                        detailsBtn.textContent = '+ Ver por mes';
                    }
                });
                
                // Añadir elementos a la celda
                ssCell.appendChild(ssValue);
                ssCell.appendChild(detailsBtn);
                ssCell.appendChild(detailsContainer);
            } else {
                // Mostrar solo el valor
                ssCell.textContent = data.safety_stock.toFixed(2);
            }
            // La variable ltCell ya fue declarada y configurada anteriormente en este bucle.
            // Celda de demanda promedio
            const avgDemandCell = document.createElement('td');
            avgDemandCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
            avgDemandCell.textContent = data.avg_demand.toFixed(2);
            
            // Celda de método
            const methodCell = document.createElement('td');
            methodCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
            
            // Mostrar el modelo utilizado si está disponible
            if (data.best_model && data.method === 'forecast') {
                const methodText = document.createElement('div');
                methodText.textContent = getMethodName(data.method);
                
                const modelText = document.createElement('div');
                modelText.className = 'text-xs text-gray-600 mt-1';
                modelText.textContent = 'Modelo: ' + data.best_model.name;
                
                methodCell.appendChild(methodText);
                methodCell.appendChild(modelText);
            } else {
                methodCell.textContent = getMethodName(data.method);
            }
            
            // Añadir celdas a la fila
            row.appendChild(articleCell);
            row.appendChild(ssCell);
            row.appendChild(ropCell);
            row.appendChild(ltCell);
            row.appendChild(avgDemandCell);
            row.appendChild(methodCell);
            
            // Añadir fila a la tabla
            tableBody.appendChild(row);
        }
    }
    
    // Función para obtener el nombre descriptivo de un método
    function getMethodName(method) {
        const methodNames = {
            'basic': 'Básico',
            'leadtime_var': 'Con Variabilidad',
            'review': 'Con Período de Revisión',
            'insufficient': 'Para Datos Insuficientes',
            'forecast': 'Con Pronósticos'
        };
        
        return methodNames[method] || method;
    }
    
    // Función para obtener el factor Z para un nivel de servicio
    function getZFactorForServiceLevel(serviceLevel) {
        // Mapa de niveles de servicio a factores Z (aproximado)
        const serviceLevelFactors = {
            0.50: 0.00,
            0.75: 0.67,
            0.80: 0.84,
            0.85: 1.04,
            0.90: 1.28,
            0.95: 1.65,
            0.96: 1.75,
            0.97: 1.88,
            0.98: 2.05,
            0.99: 2.33,
            0.995: 2.58,
            0.999: 3.09
        };
        
        // Buscar el factor Z más cercano
        let closestLevel = 0.95;  // Valor por defecto
        let minDiff = 1;
        
        for (const level in serviceLevelFactors) {
            const diff = Math.abs(parseFloat(level) - serviceLevel);
            if (diff < minDiff) {
                minDiff = diff;
                closestLevel = level;
            }
        }
        
        return serviceLevelFactors[closestLevel];
    }
    
    // Función para mostrar notificaciones
    function showNotification(message, type = 'info') {
        const notificationContainer = document.getElementById('notification-container');
        
        if (!notificationContainer) {
            // Crear contenedor de notificaciones si no existe
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'fixed top-4 right-4 z-50';
            document.body.appendChild(container);
        }
        
        // Crear notificación
        const notification = document.createElement('div');
        notification.className = `mb-3 p-3 rounded-lg shadow-md flex items-center transition-opacity duration-500 transform translate-x-0`;
        
        // Estilo según tipo
        if (type === 'success') {
            notification.classList.add('bg-green-50', 'border-l-4', 'border-green-500', 'text-green-700');
            notification.innerHTML = `<i class="fas fa-check-circle mr-2 text-green-500"></i> ${message}`;
        } else if (type === 'error') {
            notification.classList.add('bg-red-50', 'border-l-4', 'border-red-500', 'text-red-700');
            notification.innerHTML = `<i class="fas fa-exclamation-circle mr-2 text-red-500"></i> ${message}`;
        } else {
            notification.classList.add('bg-blue-50', 'border-l-4', 'border-blue-500', 'text-blue-700');
            notification.innerHTML = `<i class="fas fa-info-circle mr-2 text-blue-500"></i> ${message}`;
        }
        
        // Añadir a contenedor
        document.getElementById('notification-container').appendChild(notification);
        
        // Eliminar después de 3 segundos
        setTimeout(() => {
            notification.classList.add('opacity-0');
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 3000);
    }
    
    // Añadir opción al menú de navegación para Stock de Seguridad
    const navLinks = document.querySelectorAll('nav a');
    if (navLinks.length > 0) {
        // Buscar el último enlace de navegación para añadir después
        const lastLink = navLinks[navLinks.length - 1];
        
        // Crear nuevo enlace para Stock de Seguridad
        const newLink = document.createElement('a');
        newLink.href = '#safety-stock-section';
        newLink.className = 'text-white hover:bg-indigo-700 px-3 py-2 rounded-md text-sm font-medium flex items-center';
        newLink.innerHTML = '<i class="fas fa-shield mr-1"></i> Stock de Seguridad';
        
        // Añadir después del último enlace
        if (lastLink.parentNode) {
            lastLink.parentNode.insertBefore(newLink, lastLink.nextSibling);
        }
    }
});