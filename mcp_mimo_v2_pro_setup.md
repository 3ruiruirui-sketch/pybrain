# MIMO-V2-Pro MCP Setup Guide

## Model Overview

**MIMO-V2-Pro** is Xiaomi's flagship foundation model with impressive capabilities:
- **1T+ total parameters** with 42B active parameters (3x larger than MIMO-V2-Flash)
- **1M token context window** (1,048,576 tokens)
- **131,072 max output tokens**
- **Optimized for agentic scenarios** and OpenClaw compatibility
- **Pricing**: $1/M input tokens, $3/M output tokens

## MCP Configuration

### Current Status
- **OpenRouter MCP server**: Already available and configured
- **MIMO-V2-Pro access**: Confirmed working through OpenRouter
- **Model ID**: `xiaomi/mimo-v2-pro`

### Setup Steps

#### 1. Verify OpenRouter API Key
Ensure you have a valid OpenRouter API key with access to MIMO-V2-Pro:
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY
```

#### 2. Test MIMO-V2-Pro Connection
```python
# Using the MCP tools
from mcp0_chat_completion import chat_completion

response = chat_completion(
    messages=[{"content": "Hello, introduce yourself", "role": "user"}],
    model="xiaomi/mimo-v2-pro"
)
```

#### 3. Configure Default Model (Optional)
To set MIMO-V2-Pro as your default model in applications:

```python
# Example configuration
DEFAULT_MODEL = "xiaomi/mimo-v2-pro"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 131072  # Max for MIMO-V2-Pro
```

## Usage Examples

### Basic Chat Completion
```python
response = mcp0_chat_completion(
    messages=[
        {"content": "Explain quantum computing", "role": "user"}
    ],
    model="xiaomi/mimo-v2-pro",
    temperature=0.7,
    max_tokens=4000
)
```

### Reasoning Mode
MIMO-V2-Pro supports reasoning tokens:
```python
response = mcp0_chat_completion(
    messages=[
        {"content": "Solve this complex problem step by step", "role": "user"}
    ],
    model="xiaomi/mimo-v2-pro",
    reasoning=True  # Enable step-by-step reasoning
)
```

### Long Context Processing
With 1M token context, you can handle large documents:
```python
# Can process very long inputs
long_document = "..." # Up to 1M tokens
response = mcp0_chat_completion(
    messages=[
        {"content": long_document, "role": "user"},
        {"content": "Summarize this document", "role": "user"}
    ],
    model="xiaomi/mimo-v2-pro"
)
```

## Advanced Configuration

### Provider Settings
```python
# Configure provider-specific settings
provider_config = {
    "allow_fallbacks": True,
    "require_parameters": False,
    "sort": "latency",  # or "price", "throughput"
    "data_collection": "deny"
}
```

### Model Parameters
```python
# Optimize for different use cases
coding_params = {
    "temperature": 0.1,
    "max_tokens": 8000,
    "top_p": 0.95
}

reasoning_params = {
    "temperature": 0.5,
    "max_tokens": 16000,
    "reasoning": True
}
```

## Integration Examples

### With Existing Codebases
```python
# Replace existing model calls
# OLD: model="gpt-4"
# NEW: model="xiaomi/mimo-v2-pro"

def analyze_medical_image(image_data):
    response = mcp0_chat_completion(
        messages=[
            {"content": f"Analyze this medical image: {image_data}", "role": "user"}
        ],
        model="xiaomi/mimo-v2-pro",
        temperature=0.2  # Lower for medical accuracy
    )
    return response.choices[0].message.content
```

### For Brain Tumor Segmentation Pipeline
```python
def optimize_thresholds_with_mimo(probability_maps, uncertainty):
    """Use MIMO-V2-Pro for intelligent threshold optimization"""
    
    prompt = f"""
    Analyze these brain tumor segmentation probability maps and suggest optimal thresholds.
    
    Probability ranges:
    - Whole Tumor: {probability_maps[1].min():.3f} - {probability_maps[1].max():.3f}
    - Tumor Core: {probability_maps[0].min():.3f} - {probability_maps[0].max():.3f}
    - Enhancing: {probability_maps[2].min():.3f} - {probability_maps[2].max():.3f}
    
    Uncertainty statistics:
    - Mean: {uncertainty.mean():.3f}
    - Std: {uncertainty.std():.3f}
    - 95th percentile: {np.percentile(uncertainty, 95):.3f}
    
    Provide clinically appropriate thresholds considering:
    1. Avoid over-segmentation of edema
    2. Maintain hierarchical consistency (ET subset TC subset WT)
    3. Consider uncertainty in high-variance regions
    """
    
    response = mcp0_chat_completion(
        messages=[{"content": prompt, "role": "user"}],
        model="xiaomi/mimo-v2-pro",
        temperature=0.3,  # Lower for clinical consistency
        reasoning=True     # Enable step-by-step analysis
    )
    
    return response.choices[0].message.content
```

## Cost Management

### Token Usage Tracking
```python
def track_cost(response):
    """Track MIMO-V2-Pro usage costs"""
    usage = response['usage']
    cost = usage['cost']
    
    print(f"Tokens used: {usage['total_tokens']}")
    print(f"Cost: ${cost:.6f}")
    
    # Cost estimation
    input_cost = usage['prompt_tokens'] * 0.000001  # $1/M tokens
    output_cost = usage['completion_tokens'] * 0.000003  # $3/M tokens
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': cost
    }
```

### Budget Alerts
```python
def check_budget(usage, budget_limit=10.0):
    """Check if approaching budget limit"""
    total_cost = usage['total_cost']
    
    if total_cost > budget_limit:
        print(f"WARNING: Exceeded budget limit of ${budget_limit}")
        return False
    elif total_cost > budget_limit * 0.8:
        print(f"WARNING: Approaching budget limit (${total_cost:.2f}/${budget_limit})")
    
    return True
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Reset API key
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **Model Unavailable**
   ```python
   # Check model availability
   models = mcp0_search_models(query="xiaomi/mimo-v2-pro")
   ```

3. **Context Length Errors**
   ```python
   # Check token count before sending
   import tiktoken
   # Note: MIMO-V2-Pro may use custom tokenization
   ```

### Performance Optimization

1. **Use Caching**: Enable prompt caching for repeated queries
2. **Batch Requests**: Process multiple items in single calls
3. **Temperature Tuning**: Adjust based on task requirements
4. **Token Optimization**: Trim unnecessary context

## Next Steps

1. **Test Integration**: Run example queries with your specific use case
2. **Monitor Costs**: Track usage during initial testing phase
3. **Fine-tune Parameters**: Adjust temperature and other settings
4. **Implement Error Handling**: Add fallbacks for reliability
5. **Document Usage**: Keep track of optimal configurations

## Support Resources

- **OpenRouter Documentation**: https://openrouter.ai/docs
- **MIMO-V2-Pro Model Page**: https://openrouter.ai/xiaomi/mimo-v2-pro
- **Xiaomi MiMo Team**: Model developers and support

---

**Setup Complete!** MIMO-V2-Pro is now ready for use through your MCP configuration.
