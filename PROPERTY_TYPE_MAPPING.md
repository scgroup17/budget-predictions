# Property Type Normalization

## 🎯 Problema

El modelo ML fue entrenado con valores específicos de `property_type`, pero los usuarios pueden enviar variaciones de estos valores (ej: "Single Family" en lugar de "SFR").

## ✅ Solución

El endpoint `/predict` ahora normaliza automáticamente las variaciones comunes al formato esperado por el modelo.

## 📋 Mapeo de Valores

### Single Family Residence (SFR)

**Formato del modelo:** `SFR`

**Variaciones aceptadas:**
- `Single Family` → `SFR`
- `SingleFamily` → `SFR`
- `Single-Family` → `SFR`
- `single family` → `SFR`
- `SFR` → `SFR` (sin cambio)

### Multifamily

**Formato del modelo:** `Multifamily`

**Variaciones aceptadas:**
- `Multi Family` → `Multifamily`
- `MultiFamily` → `Multifamily`
- `Multi-Family` → `Multifamily`
- `multi family` → `Multifamily`
- `Multifamily` → `Multifamily` (sin cambio)

### Townhouse

**Formato del modelo:** `Townhouse`

**Variaciones aceptadas:**
- `Town House` → `Townhouse`
- `TownHouse` → `Townhouse`
- `town house` → `Townhouse`
- `Townhouse` → `Townhouse` (sin cambio)

### Condo

**Formato del modelo:** `Condo`

**Variaciones aceptadas:**
- `Condo` → `Condo` (sin cambio)
- `Condominium` → `Condo` (si se agrega)

## 🔧 Implementación

### Función Helper

```python
def normalize_property_type(prop_type):
    """Normalize property type variations to standard format"""
    if not prop_type:
        return 'SFR'
    
    property_type_mapping = {
        'Single Family': 'SFR',
        'SingleFamily': 'SFR',
        'Single-Family': 'SFR',
        'single family': 'SFR',
        'Multi Family': 'Multifamily',
        'MultiFamily': 'Multifamily',
        'Multi-Family': 'Multifamily',
        'multi family': 'Multifamily',
        'Town House': 'Townhouse',
        'TownHouse': 'Townhouse',
        'town house': 'Townhouse'
    }
    
    return property_type_mapping.get(prop_type, prop_type)
```

### Uso en Predict

```python
# Antes (causaba error)
prop_type = features.get('property_type', 'SFR')
encoded = LABEL_ENCODERS['Property Type'].transform([prop_type])[0]

# Después (normaliza automáticamente)
prop_type = normalize_property_type(features.get('property_type', 'SFR'))
encoded = LABEL_ENCODERS['Property Type'].transform([prop_type])[0]
```

## 📊 Ejemplos

### Ejemplo 1: Request con "Single Family"

```bash
curl -X POST https://your-api.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Plumbing Fixtures",
    "features": {
      "arv": 450000,
      "property_type": "Single Family",  # ← Será normalizado a "SFR"
      "zip_code": "33178"
    }
  }'
```

**Resultado:** ✅ Funciona correctamente

### Ejemplo 2: Request con "SFR"

```bash
curl -X POST https://your-api.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Plumbing Fixtures",
    "features": {
      "arv": 450000,
      "property_type": "SFR",  # ← Ya está en formato correcto
      "zip_code": "33178"
    }
  }'
```

**Resultado:** ✅ Funciona correctamente

### Ejemplo 3: Request con valor desconocido

```bash
curl -X POST https://your-api.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Plumbing Fixtures",
    "features": {
      "arv": 450000,
      "property_type": "Mobile Home",  # ← No está en el mapeo
      "zip_code": "33178"
    }
  }'
```

**Resultado:** 
- Se intenta usar "Mobile Home" tal cual
- Si el encoder no lo conoce, usa valor por defecto (0)
- ⚠️ Log warning: `Failed to encode property_type 'Mobile Home'. Using default.`

## 🎨 Integración en Edge Function

Tu Edge Function debe enviar el valor **tal como viene del usuario**. El endpoint Python se encarga de normalizarlo:

```typescript
// En ml-inference Edge Function
const features = {
  arv: budget.arv || 0,
  property_type: budget.property_type || 'SFR',  // ← Enviar tal cual
  zip_code: budget.zip_code || '',
  // ...
};

// Python normaliza automáticamente
```

## 🔍 Debugging

### Ver qué valor se está usando

Los logs mostrarán si hay problemas:

```
[WARNING] Failed to encode property_type 'Single Family Home': 'Single Family Home' is not in list. Using default.
```

Si ves este warning, significa que necesitas agregar esa variación al mapeo.

### Agregar Nueva Variación

En `ml_flask_service.py`, función `normalize_property_type`:

```python
property_type_mapping = {
    # ... existentes ...
    'Single Family Home': 'SFR',  # ← Agregar nueva variación
}
```

## 📋 Valores Válidos del Modelo

Los valores que el **LabelEncoder** conoce (entrenados en el modelo):

```python
# Para verificar qué valores conoce el encoder:
print(LABEL_ENCODERS['Property Type'].classes_)

# Ejemplo de output:
# ['Condo', 'Multifamily', 'SFR', 'Townhouse']
```

## 🚀 Mejoras Futuras

### V2: Normalización más inteligente

```python
def normalize_property_type(prop_type):
    """Smart normalization with fuzzy matching"""
    if not prop_type:
        return 'SFR'
    
    # Limpiar string
    clean = prop_type.strip().lower()
    
    # Fuzzy matching
    if 'single' in clean or 'sfr' in clean:
        return 'SFR'
    elif 'multi' in clean or 'apartment' in clean:
        return 'Multifamily'
    elif 'town' in clean:
        return 'Townhouse'
    elif 'condo' in clean:
        return 'Condo'
    
    # Default
    return 'SFR'
```

### V3: API que retorna valores válidos

```python
@api.route('/property-types')
class PropertyTypes(Resource):
    def get(self):
        """Get valid property types"""
        return {
            'valid_values': list(LABEL_ENCODERS['Property Type'].classes_),
            'aliases': {
                'SFR': ['Single Family', 'SingleFamily', 'Single-Family'],
                'Multifamily': ['Multi Family', 'MultiFamily', 'Multi-Family'],
                'Townhouse': ['Town House', 'TownHouse']
            }
        }
```

## ✅ Checklist

- [x] Función `normalize_property_type()` creada
- [x] Integrada en endpoint `/predict`
- [x] Manejo de errores con fallback a valor 0
- [x] Logs de warning para valores desconocidos
- [ ] Documentar en API docs (Swagger)
- [ ] Agregar tests unitarios
- [ ] Actualizar frontend con valores válidos

---

**Última actualización:** Diciembre 3, 2025
