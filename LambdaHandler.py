import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock = boto3.client(service_name='bedrock-runtime')

# Platform-to-temperature mapping
PLATFORM_TEMPERATURE_MAP = {
    # Professional platforms
    'LinkedIn': 0.3,     # More factual
    'Facebook': 0.4,     # Balanced
    'Twitter': 0.5,      # Short & catchy
    'Instagram': 0.7,    # Creative/playful
    'TikTok': 0.8,       # Highly engaging
    'Pinterest': 0.6,    # Visual emphasis
    'default': 0.5       # Fallback value
}

def get_temperature(platform):
    """Get temperature value based on platform"""
    return PLATFORM_TEMPERATURE_MAP.get(
        platform.lower().capitalize(), 
        PLATFORM_TEMPERATURE_MAP['default']
    )

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body'])
        platform = body.get('platform', 'Instagram')
        topic = body.get('topic', '')
        tone = body.get('tone', 'Friendly')
        
        # Validate input
        if not topic:
            raise ValueError("Topic cannot be empty")
            
        # Determine temperature
        temperature = get_temperature(platform)
        logger.info(f"Using temperature {temperature} for {platform}")

        # System prompt adjusted for temperature
        system_prompt = f"""You're creating {platform} posts. Guidelines:
        - Temperature setting: {temperature} ({get_temperature_description(temperature)})
        - Use {tone} tone
        - Include relevant emojis/hashtags
        - Format: [Caption] || [Hashtags]"""

        # Invoke Claude 3 Sonnet
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"Create post about: {topic}"}]
                }],
                "max_tokens": 1000,
                "temperature": temperature
            })
        )

        # Process response
        response_body = json.loads(response['body'].read())
        completion = response_body['content'][0]['text']
        caption, hashtags = completion.split('||') if '||' in completion else (completion, "")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': 'https://d3jcaby8z277ru.cloudfront.net',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'caption': caption.strip(),
                'hashtags': hashtags.strip(),
                'platform': platform,
                'temperature_used': temperature
            })
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': 'https://d3jcaby8z277ru.cloudfront.net',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'error': str(e)})
        }

def get_temperature_description(temp):
    """Helper function for system prompt"""
    if temp < 0.3: return "Highly focused"
    elif temp < 0.5: return "Professional"
    elif temp < 0.7: return "Balanced"
    else: return "Creative/Playful"
