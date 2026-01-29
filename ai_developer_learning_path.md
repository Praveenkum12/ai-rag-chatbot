# AI Developer Learning Path

## From Fullstack to AI Development

---

## Overview

This guide will help you transition from fullstack development to AI development through hands-on projects. You'll start by building on your chatbot foundation and progressively tackle different types of AI applications.

**Prerequisites:**

- Completed basic chatbot implementation
- Understanding of REST APIs
- Basic knowledge of databases
- Familiarity with async programming

**Estimated Timeline:**

- Phase 1: 3-4 weeks
- Phase 2: 4-6 weeks

---

# Phase 1: Building on Chatbot Foundation

## Project 1: RAG (Retrieval Augmented Generation)

### What You'll Build

A chatbot that can answer questions based on your company's documentation, knowledge base, or uploaded files.

### Why It Matters

RAG is one of the most important patterns in AI development. It allows AI to provide accurate, up-to-date information from specific sources rather than relying only on the model's training data.

### Learning Objectives

- ✅ Understand how embeddings represent text as vectors
- ✅ Learn document chunking strategies
- ✅ Work with vector databases
- ✅ Implement semantic search
- ✅ Combine retrieval with generation

### Step-by-Step Implementation

#### Step 1: Set Up Document Processing

```javascript
// Your tasks:
// 1. Add file upload endpoint (accept PDF, TXT, MD files)
// 2. Install document parsing library (pdf-parse for Node.js or PyPDF2 for Python)
// 3. Extract text content from uploaded files
```

**Expected Output:** A function that takes a file and returns cleaned text.

#### Step 2: Implement Text Chunking

```javascript
// Your tasks:
// 1. Split documents into chunks (500-1000 tokens each)
// 2. Add overlap between chunks (100-200 tokens)
// 3. Preserve sentence boundaries (don't cut mid-sentence)
```

**Why chunking matters:** LLMs have context limits. Smaller chunks make retrieval more precise and cost-effective.

**Tips:**

- Use libraries like `langchain` or implement custom splitter
- Experiment with chunk sizes: start with 800 tokens, 150 overlap
- Keep metadata with each chunk (source document, page number)

#### Step 3: Generate and Store Embeddings

```javascript
// Your tasks:
// 1. Sign up for OpenAI API or use open-source embedding model
// 2. Convert each chunk to embeddings using the API
// 3. Choose a vector database (start with ChromaDB or Pinecone)
// 4. Store embeddings with their original text and metadata
```

**Embedding API Example (OpenAI):**

```javascript
const response = await openai.embeddings.create({
  model: "text-embedding-3-small",
  input: "Your text chunk here",
});
const embedding = response.data[0].embedding;
```

**Vector Database Options:**

- **ChromaDB** (easiest to start, runs locally)
- **Pinecone** (managed service, free tier available)
- **Weaviate** (open-source, Docker deployment)
- **pgvector** (PostgreSQL extension, good if you already use Postgres)

#### Step 4: Implement Semantic Search

```javascript
// Your tasks:
// 1. Convert user query to embedding
// 2. Search vector database for similar chunks (top 3-5 results)
// 3. Retrieve the original text of matched chunks
```

**Search Example:**

```javascript
// User asks: "What is our refund policy?"
// 1. Convert question to embedding
// 2. Find similar document chunks
// 3. Get top 3 most relevant chunks
```

#### Step 5: Combine Retrieval with Generation

```javascript
// Your tasks:
// 1. Build a prompt that includes retrieved context
// 2. Send prompt + context + user question to LLM
// 3. Display answer with source citations
```

**Prompt Template:**

```
Context from documents:
{retrieved_chunk_1}
{retrieved_chunk_2}
{retrieved_chunk_3}

User Question: {user_question}

Instructions: Answer the question based only on the context provided above.
If the answer is not in the context, say "I don't have information about that in the documents."
Cite which document section you used.
```

### Deliverables

- [-] Document upload and processing system
- [-] Vector database setup and populated with sample docs
- [-] Search endpoint that returns relevant chunks
- [-] Enhanced chatbot that answers from your documents
- [-] Simple UI showing source citations

### Testing Checklist

- [ ] Upload a 10-page PDF and verify all pages are processed
- [ ] Ask questions that should be answerable from docs
- [ ] Ask questions NOT in docs (should say "I don't know")
- [ ] Verify citations point to correct document sections
- [ ] Test with different document types (PDF, TXT, MD)

### Bonus Challenges

- Add filters (search only specific document types or dates)
- Implement hybrid search (combine keyword + semantic search)
- Add document update/delete functionality
- Show confidence scores for retrieved results

### Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Quickstart](https://docs.pinecone.io/docs/quickstart)
- [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)

---

## Project 2: Conversation Memory and Context Management

### What You'll Build

A chatbot that remembers previous conversations and maintains coherent context across multiple turns and sessions.

### Why It Matters

Production chatbots need to handle long conversations without hitting token limits or losing context. This is crucial for user experience.

### Learning Objectives

- ✅ Understand token counting and limits
- ✅ Implement conversation history management
- ✅ Learn summarization techniques
- ✅ Handle session persistence

### Step-by-Step Implementation

#### Step 1: Implement Basic Conversation History

```javascript
// Your tasks:
// 1. Store last 10 messages in conversation array
// 2. Include both user and assistant messages
// 3. Send full history with each new request
```

**Data Structure:**

```javascript
const conversation = [
  { role: "user", content: "Hello" },
  { role: "assistant", content: "Hi! How can I help?" },
  { role: "user", content: "Tell me about AI" },
  // ... more messages
];
```

#### Step 2: Add Token Counting

```javascript
// Your tasks:
// 1. Install tiktoken or similar library
// 2. Count tokens for each message
// 3. Track total token count in conversation
// 4. Add warning when approaching limit
```

**Token Limits to Know:**

- GPT-3.5-turbo: 16,385 tokens
- GPT-4: 8,192 tokens (older) or 128,000 tokens (newer)
- GPT-4o: 128,000 tokens

**Important:** Token count includes BOTH input and output!

#### Step 3: Implement Sliding Window

```javascript
// Your tasks:
// 1. When token limit is reached, remove oldest messages
// 2. Keep system message and last N messages
// 3. Add a summary of removed messages
```

**Algorithm:**

```
1. Calculate total tokens
2. If > 75% of limit:
   - Keep system message
   - Keep last 8-10 messages
   - Summarize and remove older messages
3. Insert summary as a system message
```

#### Step 4: Add Conversation Summarization

```javascript
// Your tasks:
// 1. When removing old messages, summarize them first
// 2. Use LLM to create summary of conversation so far
// 3. Store summary and use it as context
```

**Summarization Prompt:**

```
Summarize this conversation in 2-3 sentences, preserving key facts:
{old_messages}

Focus on: important facts shared, user preferences, decisions made.
```

#### Step 5: Implement Session Persistence

```javascript
// Your tasks:
// 1. Store conversations in database with user_id and session_id
// 2. Load conversation history when user returns
// 3. Implement "start new conversation" feature
// 4. Add conversation history viewer
```

**Database Schema:**

```sql
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  user_id VARCHAR,
  session_id VARCHAR,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id),
  role VARCHAR, -- 'user' or 'assistant'
  content TEXT,
  token_count INTEGER,
  created_at TIMESTAMP
);
```

#### Step 6: Add Long-Term Memory

```javascript
// Your tasks:
// 1. Extract key facts about the user during conversations
// 2. Store these facts separately (name, preferences, history)
// 3. Inject relevant facts into new conversations
```

**Example Facts to Extract:**

- User's name, role, company
- Stated preferences ("I prefer detailed explanations")
- Previous issues or topics discussed
- Important dates or deadlines mentioned

### Deliverables

- [ ] Token counting system
- [ ] Sliding window implementation
- [ ] Conversation summarization
- [ ] Database schema and session persistence
- [ ] Conversation history UI
- [ ] Long-term memory extraction (bonus)

### Testing Checklist

- [ ] Have a 20+ turn conversation without errors
- [ ] Verify bot remembers context from 10 messages ago
- [ ] Close and reopen chat - history should persist
- [ ] Check that token limits are respected
- [ ] Test "start new conversation" feature

### Bonus Challenges

- Implement conversation search (find past conversations by topic)
- Add conversation export (download as PDF or text)
- Create conversation analytics (avg length, topics discussed)
- Implement "remember this" command for explicit fact storage

---

## Project 3: Function Calling / Tool Use

### What You'll Build

A chatbot that can perform actions like querying databases, calling APIs, or triggering workflows based on user requests.

### Why It Matters

This transforms your chatbot from a simple Q&A system into an intelligent agent that can actually do things for users.

### Learning Objectives

- ✅ Understand LLM function calling capabilities
- ✅ Define function schemas properly
- ✅ Handle function execution flow
- ✅ Manage errors gracefully
- ✅ Chain multiple function calls

### Step-by-Step Implementation

#### Step 1: Define Your First Function

```javascript
// Your tasks:
// 1. Choose a simple function (e.g., get current weather)
// 2. Define the function schema in OpenAI format
// 3. Implement the actual function logic
```

**Function Schema Example:**

```javascript
const tools = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "City name, e.g., San Francisco",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "Temperature unit",
          },
        },
        required: ["location"],
      },
    },
  },
];
```

**Actual Function:**

```javascript
async function get_weather(location, unit = "fahrenheit") {
  // Call weather API
  const response = await fetch(`https://api.weather.com/...`);
  return response.json();
}
```

#### Step 2: Implement Function Calling Flow

```javascript
// Your tasks:
// 1. Send user message with available tools to LLM
// 2. Check if LLM wants to call a function
// 3. Execute the function with provided parameters
// 4. Send function result back to LLM
// 5. Return final response to user
```

**Flow Diagram:**

```
User: "What's the weather in New York?"
   ↓
1. Send to LLM with function definitions
   ↓
2. LLM responds: "Call get_weather(location='New York')"
   ↓
3. Execute get_weather("New York")
   ↓
4. Get result: {temp: 72, condition: "sunny"}
   ↓
5. Send result back to LLM
   ↓
6. LLM responds: "It's 72°F and sunny in New York"
   ↓
User receives friendly response
```

**Code Structure:**

```javascript
// 1. Initial request
const response = await openai.chat.completions.create({
  model: "gpt-4",
  messages: messages,
  tools: tools,
});

// 2. Check for function call
if (response.choices[0].finish_reason === "tool_calls") {
  const toolCall = response.choices[0].message.tool_calls[0];

  // 3. Execute function
  const functionResult = await executeFunct(
    toolCall.function.name,
    JSON.parse(toolCall.function.arguments),
  );

  // 4. Send result back
  messages.push(response.choices[0].message);
  messages.push({
    role: "tool",
    tool_call_id: toolCall.id,
    content: JSON.stringify(functionResult),
  });

  // 5. Get final response
  const finalResponse = await openai.chat.completions.create({
    model: "gpt-4",
    messages: messages,
  });
}
```

#### Step 3: Add More Useful Functions

```javascript
// Your tasks:
// 1. Add database query function
// 2. Add task creation function
// 3. Add web search function (if available)
// 4. Test each function individually
```

**Suggested Functions:**

```javascript
// Database query
async function search_tickets(status, assignee) {
  // Query your database
  return await db.query("SELECT * FROM tickets WHERE ...");
}

// Task creation
async function create_task(title, description, assignee, due_date) {
  // Call project management API
  return await projectAPI.createTask({...});
}

// Send email
async function send_email(to, subject, body) {
  // Call email service
  return await emailService.send({...});
}

// Calculator
function calculate(expression) {
  // Safe math evaluation
  return math.evaluate(expression);
}
```

#### Step 4: Handle Errors and Edge Cases

```javascript
// Your tasks:
// 1. Add try-catch around function execution
// 2. Handle missing parameters
// 3. Validate function inputs
// 4. Return meaningful error messages
```

**Error Handling:**

```javascript
async function executeFunction(name, args) {
  try {
    // Validate required parameters
    if (name === "get_weather" && !args.location) {
      return { error: "Location is required" };
    }

    // Execute function
    const result = await functions[name](args);

    // Validate result
    if (!result) {
      return { error: "Function returned no data" };
    }

    return result;
  } catch (error) {
    console.error(`Function ${name} failed:`, error);
    return {
      error: `Failed to execute ${name}: ${error.message}`,
    };
  }
}
```

#### Step 5: Implement Function Chaining

```javascript
// Your tasks:
// 1. Handle cases where LLM needs multiple function calls
// 2. Implement loop to keep calling until no more functions needed
// 3. Add safeguard against infinite loops (max 5 iterations)
```

**Example Multi-Step Query:**

```
User: "Email my team the weather forecast for tomorrow's meeting location"

Step 1: get_meeting_location() → "San Francisco"
Step 2: get_weather("San Francisco") → "Sunny, 75°F"
Step 3: get_team_emails() → ["john@...", "jane@..."]
Step 4: send_email(to=team, subject="Weather", body="...")
```

#### Step 6: Add Security and Permissions

```javascript
// Your tasks:
// 1. Create allowlist of safe functions
// 2. Add user permission checks
// 3. Add confirmation for destructive actions
// 4. Log all function calls for audit
```

**Security Checklist:**

- [ ] Never allow arbitrary code execution
- [ ] Validate all function parameters
- [ ] Require confirmation for: delete, send_email, create_payment
- [ ] Rate limit function calls
- [ ] Log all function executions with user_id and timestamp

### Deliverables

- [ ] At least 3 working functions with proper schemas
- [ ] Complete function calling flow implementation
- [ ] Error handling for all edge cases
- [ ] Function chaining support
- [ ] Security and permission system
- [ ] Logging/audit trail

### Testing Checklist

- [ ] Test each function individually
- [ ] Test successful function calls through chatbot
- [ ] Test with invalid parameters
- [ ] Test function chaining (2+ functions in sequence)
- [ ] Test permission denials
- [ ] Test rate limiting

### Bonus Challenges

- Add function call confirmation UI ("Bot wants to send email. Approve?")
- Implement function call history viewer
- Add function call analytics (most used functions, success rates)
- Create custom functions specific to your company's needs
- Implement parallel function execution where possible

### Real-World Function Ideas

**For your company:**

- Query customer data
- Create support tickets
- Schedule meetings
- Generate reports
- Update CRM records
- Search internal docs
- Check system status
- Trigger deployments (with proper safeguards!)

---

# Phase 2: Different AI Application Types

## Project 4: Image Generation or Manipulation Tool

### What You'll Build

An application that creates or modifies images using AI, such as a marketing asset generator or product mockup creator.

### Why It Matters

Image AI is one of the fastest-growing areas in AI development. Understanding how to work with image models opens up creative possibilities for marketing, design, and content creation.

### Learning Objectives

- ✅ Work with image generation APIs
- ✅ Master prompt engineering for images
- ✅ Handle image uploads and processing
- ✅ Understand model parameters and their effects
- ✅ Optimize costs and performance

### Choose Your Project Track

Pick ONE of these to implement first:

**Track A: Simple Image Generator**

- Text-to-image generation
- Style selection (photorealistic, cartoon, sketch)
- Image variations

**Track B: Marketing Asset Generator**

- Generate social media images
- Create blog headers
- Product mockups with text overlays

**Track C: Image Editor Bot**

- Upload image + describe edits
- Background removal/replacement
- Object addition/removal

### Step-by-Step Implementation

#### Step 1: Set Up Image API

```javascript
// Your tasks:
// 1. Choose your API (OpenAI DALL-E, Stability AI, or Replicate)
// 2. Sign up and get API key
// 3. Install SDK
// 4. Test basic image generation
```

**OpenAI DALL-E Example:**

```javascript
const response = await openai.images.generate({
  model: "dall-e-3",
  prompt: "A professional headshot of a businesswoman in an office",
  n: 1,
  size: "1024x1024",
});

const imageUrl = response.data[0].url;
```

**API Comparison:**

- **DALL-E 3**: Best quality, higher cost ($0.040/image)
- **DALL-E 2**: Good quality, lower cost ($0.020/image)
- **Stability AI**: More control, cheapest (~$0.002/image)

#### Step 2: Build Basic Image Generation UI

```javascript
// Your tasks:
// 1. Create form with prompt textarea
// 2. Add size selection (1024x1024, 1792x1024, etc.)
// 3. Add "Generate" button
// 4. Display loading state
// 5. Show generated image with download button
```

**UI Requirements:**

- Clear prompt input (with character counter)
- Size/aspect ratio selector
- Style selector (if applicable)
- Loading indicator with estimated time
- Image preview with zoom
- Download button
- Regenerate option

#### Step 3: Implement Prompt Engineering

```javascript
// Your tasks:
// 1. Research effective image prompts
// 2. Add prompt templates/examples
// 3. Implement prompt enhancement (auto-add quality modifiers)
// 4. Add negative prompts support (if using Stability AI)
```

**Prompt Engineering Tips:**

**Structure of a good prompt:**

```
[Subject] [Action] [Context/Setting] [Style] [Quality modifiers]

Example:
"A golden retriever playing fetch in a sunny park,
watercolor painting style, soft lighting,
high quality, detailed"
```

**Quality Modifiers:**

- For realistic: "photorealistic, 8k, high detail, professional photography"
- For art: "digital art, trending on artstation, highly detailed"
- For corporate: "professional, clean, modern, high quality"

**Negative Prompts (for Stability AI):**

```
"blurry, low quality, distorted, disfigured, bad anatomy"
```

**Prompt Templates:**

```javascript
const templates = {
  productPhoto:
    "Professional product photography of {product}, white background, studio lighting, high resolution, commercial quality",

  socialMedia:
    "Eye-catching social media post image for {topic}, vibrant colors, modern design, {style} style",

  blogHeader:
    "Hero image for blog post about {topic}, {mood} mood, horizontal layout, professional quality",

  avatar:
    "Professional {style} portrait, {description}, clean background, centered composition",
};
```

#### Step 4: Add Image Storage and Management

```javascript
// Your tasks:
// 1. Set up cloud storage (S3, Cloudinary, etc.)
// 2. Download and store generated images
// 3. Save metadata (prompt, parameters, user_id)
// 4. Create image gallery view
// 5. Add search/filter functionality
```

**Storage Schema:**

```sql
CREATE TABLE generated_images (
  id UUID PRIMARY KEY,
  user_id VARCHAR,
  prompt TEXT,
  negative_prompt TEXT,
  model VARCHAR,
  size VARCHAR,
  style VARCHAR,
  image_url TEXT,
  thumbnail_url TEXT,
  created_at TIMESTAMP
);
```

#### Step 5: Implement Advanced Features (Choose 2-3)

**Option A: Image Variations**

```javascript
// User uploads an image, generate similar versions
const response = await openai.images.createVariation({
  image: fs.createReadStream("original.png"),
  n: 3,
  size: "1024x1024",
});
```

**Option B: Image Editing**

```javascript
// User uploads image + mask, describes edit
const response = await openai.images.edit({
  image: fs.createReadStream("original.png"),
  mask: fs.createReadStream("mask.png"),
  prompt: "A sunflower in the center",
  n: 1,
  size: "1024x1024",
});
```

**Option C: Batch Generation**

```javascript
// Generate multiple images from a list of prompts
// Useful for: A/B testing, bulk content creation
```

**Option D: Style Transfer**

```javascript
// Apply a specific art style to uploaded images
// Requires: img2img with style reference
```

**Option E: Text Overlay**

```javascript
// Add text to generated images
// Use: Canvas API or image processing library
const canvas = createCanvas(1024, 1024);
const ctx = canvas.getContext("2d");
// Load image, add text, export
```

#### Step 6: Optimize for Production

```javascript
// Your tasks:
// 1. Add request queuing for batch jobs
// 2. Implement caching (same prompt = same image)
// 3. Add cost tracking per user
// 4. Implement rate limiting
// 5. Add image moderation/safety checks
```

**Cost Optimization:**

```javascript
// Cache prompts to avoid duplicate generations
const cacheKey = `img:${hash(prompt + size + style)}`;
const cached = await cache.get(cacheKey);
if (cached) return cached;

// Track spending
await db.trackCost({
  user_id,
  service: "dalle-3",
  cost: 0.04,
  timestamp: new Date(),
});
```

**Safety Checks:**

```javascript
// Screen prompts before sending to API
const isSafe = await moderateContent(prompt);
if (!isSafe) {
  throw new Error("Prompt violates content policy");
}

// Screen generated images (if required)
const imageModeration = await moderateImage(imageUrl);
```

### Deliverables

- [ ] Working image generation interface
- [ ] At least 5 prompt templates
- [ ] Image storage system
- [ ] User gallery with search
- [ ] 2-3 advanced features implemented
- [ ] Cost tracking dashboard
- [ ] Documentation of your prompt engineering discoveries

### Testing Checklist

- [ ] Generate images with various prompts
- [ ] Test different sizes and styles
- [ ] Verify images are saved correctly
- [ ] Check cost tracking accuracy
- [ ] Test with edge case prompts
- [ ] Verify rate limiting works
- [ ] Test on mobile devices

### Bonus Challenges

- Implement A/B testing for prompts (generate 2 versions, let user pick)
- Add collaborative features (share prompts, like images)
- Create prompt library with community contributions
- Implement NSFW content filtering
- Add image upscaling for small images
- Build Chrome extension for generating images from any webpage
- Create API for programmatic access

### Real-World Use Cases to Explore

- E-commerce product mockups
- Social media content calendar automation
- Blog illustration pipeline
- Presentation slide backgrounds
- Marketing campaign assets
- Profile picture generator
- Concept art for projects

---

## Project 5: Document Processing System

### What You'll Build

An intelligent system that extracts, analyzes, and structures information from documents automatically.

### Why It Matters

Most business information is trapped in unstructured documents. Building systems that can process these at scale is incredibly valuable.

### Learning Objectives

- ✅ Parse various document formats (PDF, DOCX, images)
- ✅ Extract structured data from unstructured text
- ✅ Implement OCR for scanned documents
- ✅ Handle large documents efficiently
- ✅ Validate and structure AI outputs

### Choose Your Project Track

Pick ONE to implement first:

**Track A: Invoice/Receipt Processor**

- Extract: vendor, date, items, amounts, tax, total
- Output: Structured JSON
- Use case: Expense management

**Track B: Resume Parser**

- Extract: contact info, experience, education, skills
- Output: Structured profile
- Use case: Recruitment automation

**Track C: Contract Analyzer**

- Extract: parties, dates, key terms, obligations
- Output: Summary + structured data
- Use case: Legal review

**Track D: Meeting Notes Processor**

- Extract: attendees, decisions, action items, deadlines
- Output: Organized notes
- Use case: Team productivity

### Step-by-Step Implementation

#### Step 1: Set Up Document Parsing

```javascript
// Your tasks:
// 1. Install document parsing libraries
// 2. Implement PDF text extraction
// 3. Implement DOCX parsing
// 4. Handle images (JPG, PNG)
```

**Library Recommendations:**

**Node.js:**

```bash
npm install pdf-parse mammoth tesseract.js
```

**Python:**

```bash
pip install PyPDF2 python-docx pytesseract Pillow
```

**Basic PDF Parser:**

```javascript
const pdf = require("pdf-parse");
const fs = require("fs");

async function extractTextFromPDF(filepath) {
  const dataBuffer = fs.readFileSync(filepath);
  const data = await pdf(dataBuffer);
  return {
    text: data.text,
    pages: data.numpages,
    info: data.info,
  };
}
```

**Basic DOCX Parser:**

```javascript
const mammoth = require("mammoth");

async function extractTextFromDOCX(filepath) {
  const result = await mammoth.extractRawText({ path: filepath });
  return result.value;
}
```

#### Step 2: Implement OCR for Scanned Documents

```javascript
// Your tasks:
// 1. Install Tesseract OCR
// 2. Implement image-to-text extraction
// 3. Add preprocessing (deskew, denoise)
// 4. Handle multi-page scanned PDFs
```

**OCR Implementation:**

```javascript
const Tesseract = require("tesseract.js");

async function performOCR(imagePath) {
  const {
    data: { text },
  } = await Tesseract.recognize(imagePath, "eng", {
    logger: (info) => console.log(info), // Progress tracking
  });
  return text;
}
```

**Image Preprocessing (improves OCR accuracy):**

```python
from PIL import Image, ImageFilter, ImageEnhance

def preprocess_image(image_path):
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)

    # Denoise
    img = img.filter(ImageFilter.MedianFilter())

    return img
```

#### Step 3: Design Extraction Schema

```javascript
// Your tasks:
// 1. Define what data you want to extract
// 2. Create JSON schema for output
// 3. Design validation rules
```

**Example Schema for Invoice:**

```javascript
const invoiceSchema = {
  vendor: {
    name: "string",
    address: "string",
    phone: "string",
    email: "string",
  },
  invoice_number: "string",
  date: "date",
  due_date: "date",
  items: [
    {
      description: "string",
      quantity: "number",
      unit_price: "number",
      total: "number",
    },
  ],
  subtotal: "number",
  tax: "number",
  total: "number",
  payment_terms: "string",
};
```

**Example Schema for Resume:**

```javascript
const resumeSchema = {
  personal: {
    name: "string",
    email: "string",
    phone: "string",
    location: "string",
    linkedin: "string",
    website: "string",
  },
  summary: "string",
  experience: [
    {
      company: "string",
      position: "string",
      start_date: "date",
      end_date: "date | 'Present'",
      description: "string",
      achievements: ["string"],
    },
  ],
  education: [
    {
      institution: "string",
      degree: "string",
      field: "string",
      graduation_date: "date",
      gpa: "number (optional)",
    },
  ],
  skills: {
    technical: ["string"],
    soft: ["string"],
    languages: ["string"],
  },
  certifications: ["string"],
};
```

#### Step 4: Implement AI Extraction

```javascript
// Your tasks:
// 1. Create extraction prompt with schema
// 2. Send document text + schema to LLM
// 3. Request JSON output
// 4. Parse and validate response
```

**Extraction Prompt Template:**

```javascript
const extractionPrompt = `
You are a document data extraction system.

Extract the following information from the document below and return ONLY valid JSON.

Required Schema:
${JSON.stringify(schema, null, 2)}

Document Text:
${documentText}

Instructions:
1. Extract all information that matches the schema
2. If information is missing, use null
3. Ensure dates are in YYYY-MM-DD format
4. Ensure numbers are numeric types, not strings
5. Return ONLY the JSON object, no other text

JSON Output:
`;
```

**Making the API Call:**

```javascript
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [
    {
      role: "system",
      content: "You are a data extraction expert. Return only valid JSON.",
    },
    {
      role: "user",
      content: extractionPrompt,
    },
  ],
  response_format: { type: "json_object" }, // Forces JSON output
});

const extracted = JSON.parse(response.choices[0].message.content);
```

#### Step 5: Validate and Structure Output

```javascript
// Your tasks:
// 1. Implement schema validation
// 2. Check for required fields
// 3. Validate data types and formats
// 4. Add confidence scoring
// 5. Flag incomplete extractions
```

**Validation Example:**

```javascript
function validateExtraction(data, schema) {
  const errors = [];
  const warnings = [];

  // Check required fields
  if (!data.invoice_number) {
    errors.push("Missing invoice number");
  }

  // Validate data types
  if (data.total && typeof data.total !== "number") {
    errors.push("Total must be a number");
  }

  // Validate formats
  if (data.date && !isValidDate(data.date)) {
    errors.push("Invalid date format");
  }

  // Check logical consistency
  if (data.subtotal && data.total && data.total < data.subtotal) {
    warnings.push("Total is less than subtotal");
  }

  // Calculate completeness
  const totalFields = countFields(schema);
  const filledFields = countFilledFields(data);
  const completeness = (filledFields / totalFields) * 100;

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    completeness: `${completeness.toFixed(1)}%`,
  };
}
```

#### Step 6: Build Document Processing Pipeline

```javascript
// Your tasks:
// 1. Create upload endpoint
// 2. Detect document type
// 3. Route to appropriate processor
// 4. Store results
// 5. Return structured data
```

**Pipeline Flow:**

```
1. Upload Document
   ↓
2. Detect Type (PDF, DOCX, Image)
   ↓
3. Extract Text
   ↓ (if scanned)
4. Perform OCR
   ↓
5. Send to AI for Extraction
   ↓
6. Validate Output
   ↓
7. Store in Database
   ↓
8. Return Structured Data
```

**Pipeline Code:**

```javascript
async function processDocument(file, documentType) {
  try {
    // Step 1: Extract text
    let text;
    if (file.mimetype === "application/pdf") {
      const pdfData = await extractTextFromPDF(file.path);
      text = pdfData.text;

      // Check if it's a scanned PDF (very little text)
      if (text.length < 100) {
        text = await performOCR(file.path);
      }
    } else if (file.mimetype.includes("image")) {
      text = await performOCR(file.path);
    } else if (file.mimetype.includes("word")) {
      text = await extractTextFromDOCX(file.path);
    }

    // Step 2: Get schema for document type
    const schema = getSchemaForType(documentType);

    // Step 3: Extract data
    const extracted = await extractData(text, schema);

    // Step 4: Validate
    const validation = validateExtraction(extracted, schema);

    // Step 5: Store
    const result = await db.storeExtraction({
      filename: file.originalname,
      document_type: documentType,
      extracted_data: extracted,
      validation,
      raw_text: text,
      processed_at: new Date(),
    });

    return {
      id: result.id,
      data: extracted,
      validation,
      success: validation.valid,
    };
  } catch (error) {
    console.error("Document processing failed:", error);
    throw error;
  }
}
```

#### Step 7: Create Review Interface

```javascript
// Your tasks:
// 1. Build UI to display extracted data
// 2. Show confidence scores
// 3. Allow manual corrections
// 4. Display original document alongside
// 5. Add approval/rejection workflow
```

**UI Features:**

- Split screen: original document on left, extracted data on right
- Highlight extracted fields in original document
- Color-code confidence levels (green = high, yellow = medium, red = low)
- Inline editing for corrections
- Approve/Reject buttons
- Export to CSV/JSON

### Deliverables

- [ ] Multi-format document parser (PDF, DOCX, images)
- [ ] OCR implementation for scanned docs
- [ ] AI extraction with your chosen schema
- [ ] Validation system
- [ ] Complete processing pipeline
- [ ] Review/correction interface
- [ ] Export functionality

### Testing Checklist

- [ ] Test with 10+ sample documents
- [ ] Test with scanned/poor quality documents
- [ ] Verify extraction accuracy (>90% target)
- [ ] Test validation catches errors
- [ ] Check processing time (should be <30s per doc)
- [ ] Test with edge cases (missing fields, unusual formats)
- [ ] Verify exports work correctly

### Bonus Challenges

- Implement table extraction from PDFs
- Add multi-language support
- Create batch processing for folders of documents
- Build confidence scoring for each field
- Add learning from corrections (store feedback)
- Implement document classification (auto-detect type)
- Create comparison tool (compare 2 versions of a document)
- Add email integration (process attachments automatically)

### Real-World Applications

- **Accounting:** Automate invoice processing
- **HR:** Parse resumes at scale
- **Legal:** Extract contract terms
- **Healthcare:** Process medical records
- **Real Estate:** Parse property documents
- **Education:** Grade and analyze essays
- **Finance:** Process bank statements

### Performance Optimization Tips

```javascript
// Chunk large documents
if (text.length > 10000) {
  const chunks = splitIntoChunks(text, 8000);
  const results = await Promise.all(
    chunks.map((chunk) => extractData(chunk, schema)),
  );
  return mergeResults(results);
}

// Cache common extractions
const cacheKey = `doc:${hash(text)}:${documentType}`;
const cached = await cache.get(cacheKey);
if (cached) return cached;

// Use cheaper models for simple extractions
const model = complexity === "simple" ? "gpt-3.5-turbo" : "gpt-4o";
```

---

## Project 6: Code Review Assistant

### What You'll Build

An AI-powered tool that analyzes code changes and provides intelligent feedback on quality, bugs, security issues, and best practices.

### Why It Matters

Code review is time-consuming but critical. AI can catch common issues, enforce standards, and help less experienced developers learn best practices.

### Learning Objectives

- ✅ Parse and understand Git diffs
- ✅ Analyze code with LLMs
- ✅ Provide actionable technical feedback
- ✅ Integrate with version control platforms
- ✅ Balance automation with developer experience

### Choose Your Project Track

Pick ONE to implement first:

**Track A: PR Review Bot**

- Integrates with GitHub/GitLab
- Comments on pull requests
- Full automation

**Track B: Code Quality CLI**

- Command-line tool
- Run before committing
- Local feedback

**Track C: Code Explainer**

- Paste code, get explanation
- Good for learning
- Simpler to build

### Step-by-Step Implementation

#### Step 1: Parse Git Diffs

```javascript
// Your tasks:
// 1. Install git diff parser library
// 2. Parse diff format into structured data
// 3. Extract changed files and lines
// 4. Get surrounding context
```

**Understanding Git Diff Format:**

```diff
diff --git a/src/utils.js b/src/utils.js
index 1234567..abcdefg 100644
--- a/src/utils.js
+++ b/src/utils.js
@@ -10,7 +10,7 @@ function calculateTotal(items) {
   let total = 0;
   for (let item of items) {
-    total += item.price;
+    total += item.price * item.quantity;
   }
   return total;
 }
```

**Parsing Code:**

```javascript
const parseDiff = require("parse-diff");

function parseGitDiff(diffString) {
  const files = parseDiff(diffString);

  return files.map((file) => ({
    filename: file.to,
    additions: file.additions,
    deletions: file.deletions,
    chunks: file.chunks.map((chunk) => ({
      oldStart: chunk.oldStart,
      newStart: chunk.newStart,
      changes: chunk.changes.map((change) => ({
        type: change.type, // 'add', 'del', 'normal'
        content: change.content,
        lineNumber: change.ln || change.ln1,
      })),
    })),
  }));
}
```

#### Step 2: Get File Context

```javascript
// Your tasks:
// 1. Fetch the full file (not just the diff)
// 2. Get surrounding lines for context
// 3. Include imports and function signatures
```

**Why context matters:**

```javascript
// Without context, this looks fine:
+ const result = calculateTotal(items);

// With context, we see it's inside a loop (bad!):
for (let i = 0; i < users.length; i++) {
+   const result = calculateTotal(items);
}
```

**Getting Context:**

```javascript
async function getFileContext(repo, filepath, startLine, endLine) {
  // Get full file content
  const file = await github.repos.getContent({
    owner: repo.owner,
    repo: repo.name,
    path: filepath,
    ref: "main",
  });

  const content = Buffer.from(file.data.content, "base64").toString();
  const lines = content.split("\n");

  // Get context (e.g., 10 lines before and after)
  const contextStart = Math.max(0, startLine - 10);
  const contextEnd = Math.min(lines.length, endLine + 10);

  return {
    context: lines.slice(contextStart, contextEnd).join("\n"),
    fullFile: content,
  };
}
```

#### Step 3: Design Review Prompts

```javascript
// Your tasks:
// 1. Create system prompt for code reviewer
// 2. Structure review request with context
// 3. Define review categories
// 4. Request structured output format
```

**System Prompt:**

```javascript
const SYSTEM_PROMPT = `
You are an expert code reviewer. Analyze the provided code changes and provide constructive feedback.

Focus on:
1. **Bugs**: Logic errors, null pointer exceptions, off-by-one errors
2. **Security**: SQL injection, XSS, insecure dependencies, exposed secrets
3. **Performance**: Inefficient algorithms, unnecessary loops, memory leaks
4. **Best Practices**: Code style, naming, structure, SOLID principles
5. **Readability**: Complex logic, missing comments, unclear variable names

For each issue found:
- Specify severity: critical, warning, or suggestion
- Explain the problem clearly
- Provide specific line numbers
- Suggest a fix with code example if applicable

Be constructive and encouraging. Acknowledge good code too.
`;
```

**Review Request Format:**

```javascript
const reviewPrompt = `
File: ${filename}
Language: ${language}

Changed Code:
\`\`\`${language}
${changedCode}
\`\`\`

Full Context:
\`\`\`${language}
${fullContext}
\`\`\`

Please review these changes and return your feedback in JSON format:
{
  "issues": [
    {
      "severity": "critical|warning|suggestion",
      "category": "bug|security|performance|style|readability",
      "line": <line_number>,
      "message": "Description of the issue",
      "suggestion": "Recommended fix (optional)"
    }
  ],
  "positives": ["Things done well"],
  "overall_score": <1-10>
}
`;
```

#### Step 4: Implement Core Review Logic

```javascript
// Your tasks:
// 1. Send code + context to LLM
// 2. Parse structured feedback
// 3. Filter false positives
// 4. Prioritize issues by severity
```

**Review Function:**

```javascript
async function reviewCode(diff, context) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: createReviewPrompt(diff, context) },
    ],
    response_format: { type: "json_object" },
  });

  const review = JSON.parse(response.choices[0].message.content);

  // Filter and prioritize
  const filteredIssues = review.issues
    .filter((issue) => !isFalsePositive(issue))
    .sort((a, b) => severityWeight(b.severity) - severityWeight(a.severity));

  return {
    ...review,
    issues: filteredIssues,
  };
}

function severityWeight(severity) {
  return { critical: 3, warning: 2, suggestion: 1 }[severity] || 0;
}
```

#### Step 5: Integrate with GitHub/GitLab

```javascript
// Your tasks:
// 1. Set up webhook for PR events
// 2. Fetch PR diff when opened/updated
// 3. Run review
// 4. Post comments on PR
```

**GitHub Integration:**

```javascript
const { Octokit } = require("@octokit/rest");
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

// Webhook handler
app.post("/webhook/github", async (req, res) => {
  const event = req.body;

  if (event.action === "opened" || event.action === "synchronize") {
    const pr = event.pull_request;

    // Get PR diff
    const diff = await octokit.pulls.get({
      owner: event.repository.owner.login,
      repo: event.repository.name,
      pull_number: pr.number,
      mediaType: { format: "diff" },
    });

    // Review the changes
    const review = await reviewCode(diff.data);

    // Post review comments
    await postReviewComments(event.repository, pr.number, review);
  }

  res.sendStatus(200);
});

async function postReviewComments(repo, prNumber, review) {
  // Post overall summary
  await octokit.issues.createComment({
    owner: repo.owner.login,
    repo: repo.name,
    issue_number: prNumber,
    body: formatSummary(review),
  });

  // Post inline comments for specific issues
  for (const issue of review.issues) {
    await octokit.pulls.createReviewComment({
      owner: repo.owner.login,
      repo: repo.name,
      pull_number: prNumber,
      body: formatIssueComment(issue),
      path: issue.filename,
      line: issue.line,
      side: "RIGHT",
    });
  }
}
```

#### Step 6: Add Advanced Features

```javascript
// Your tasks (choose 2-3):
// 1. Security-focused review mode
// 2. Language-specific checks
// 3. Custom rule configuration
// 4. Learn from review feedback
```

**Security-Focused Mode:**

```javascript
const SECURITY_PROMPT = `
Focus ONLY on security issues:
- SQL injection vulnerabilities
- XSS and CSRF risks
- Hardcoded credentials or API keys
- Insecure dependencies
- Authentication/authorization flaws
- Sensitive data exposure
- Input validation issues

For each security issue:
- Rate severity: CRITICAL, HIGH, MEDIUM, LOW
- Explain the attack vector
- Provide secure code example
`;
```

**Language-Specific Checks:**

```javascript
const LANGUAGE_RULES = {
  javascript: [
    "Check for == vs ===",
    "Verify async/await error handling",
    "Look for potential memory leaks in event listeners",
    "Check for proper input sanitization",
  ],
  python: [
    "Check for SQL injection in string formatting",
    "Verify exception handling",
    "Look for mutable default arguments",
    "Check for proper resource cleanup (context managers)",
  ],
  // Add more languages...
};
```

**Custom Configuration:**

```javascript
// .codereview.json in repo
{
  "enabled": true,
  "severity_threshold": "warning",
  "ignore_patterns": ["*.test.js", "migrations/*"],
  "custom_rules": [
    {
      "name": "No console.log in production",
      "pattern": "console\\.log",
      "severity": "warning",
      "message": "Remove console.log before merging"
    }
  ],
  "max_issues_per_review": 10
}
```

### Deliverables

- [ ] Diff parser working for your language
- [ ] Code review logic with structured output
- [ ] At least 5 review categories implemented
- [ ] GitHub/GitLab integration OR CLI tool
- [ ] Comment posting functionality
- [ ] Configuration system
- [ ] Documentation for setup

### Testing Checklist

- [ ] Test with simple bug (e.g., null pointer)
- [ ] Test with security issue (e.g., SQL injection)
- [ ] Test with style issue (e.g., naming)
- [ ] Test with correct code (should have no/few issues)
- [ ] Verify inline comments appear on correct lines
- [ ] Check false positive rate (should be <20%)
- [ ] Test with large PRs (100+ line changes)

### Bonus Challenges

- Add auto-fix suggestions (generate corrected code)
- Implement "explain this code" feature
- Create diff between suggested fix and original
- Add code complexity metrics
- Build dashboard showing review trends
- Implement learning from accepted/rejected suggestions
- Add team-specific customization (learn team's style)
- Support multiple programming languages

### Real-World Use Cases

- **Onboarding**: Help new developers learn best practices
- **Security**: Catch vulnerabilities before production
- **Consistency**: Enforce team coding standards
- **Education**: Explain why something is wrong
- **Time-saving**: Focus human reviewers on complex logic

### Anti-Patterns to Avoid

```javascript
// ❌ Don't be too noisy
// Posting 50 minor suggestions overwhelms developers

// ✅ Prioritize and limit
const topIssues = issues
  .filter((i) => i.severity !== "suggestion")
  .slice(0, 10);

// ❌ Don't block for style issues
if (hasOnlyStyleIssues) {
  approve = true;
  postSuggestions = true;
}

// ✅ Auto-approve if only minor issues
// ❌ Don't be overly prescriptive
("Your variable name should be 'userAuthenticationToken'");

// ✅ Be flexible
("Consider a more descriptive name than 'x' for better readability");
```

---

## Project 7: Sentiment Analysis Dashboard

### What You'll Build

A system that analyzes customer feedback at scale, extracts insights, and visualizes trends over time.

### Why It Matters

Companies collect tons of feedback but struggle to make sense of it. Automated sentiment analysis turns unstructured feedback into actionable insights.

### Learning Objectives

- ✅ Implement text classification and analysis
- ✅ Work with batch processing
- ✅ Create meaningful data visualizations
- ✅ Build aggregation and analytics pipelines
- ✅ Handle real-time vs batch analysis

### Choose Your Project Track

Pick ONE to implement first:

**Track A: Customer Support Analyzer**

- Analyze support tickets
- Detect urgent/angry customers
- Track resolution satisfaction

**Track B: Product Review Dashboard**

- Analyze product reviews
- Extract feature feedback
- Compare sentiment across products

**Track C: Social Media Monitor**

- Track brand mentions
- Alert on negative sentiment spikes
- Competitive analysis

**Track D: Survey Response Analyzer**

- Process open-ended survey responses
- Extract themes and patterns
- Generate insights report

### Step-by-Step Implementation

#### Step 1: Design Your Data Model

```javascript
// Your tasks:
// 1. Choose your feedback source (tickets, reviews, surveys)
// 2. Design database schema
// 3. Plan what metrics to track
```

**Database Schema:**

```sql
-- Original feedback
CREATE TABLE feedback (
  id UUID PRIMARY KEY,
  source VARCHAR(50), -- 'support_ticket', 'review', 'survey'
  source_id VARCHAR,
  user_id VARCHAR,
  text TEXT,
  metadata JSONB, -- product_id, category, etc.
  created_at TIMESTAMP,
  processed_at TIMESTAMP
);

-- Analysis results
CREATE TABLE sentiment_analysis (
  id UUID PRIMARY KEY,
  feedback_id UUID REFERENCES feedback(id),
  sentiment VARCHAR(20), -- 'positive', 'negative', 'neutral', 'mixed'
  sentiment_score DECIMAL(3,2), -- -1.0 to 1.0
  confidence DECIMAL(3,2), -- 0.0 to 1.0
  emotions JSONB, -- { "joy": 0.8, "anger": 0.1, ... }
  topics JSONB, -- ["pricing", "customer_service"]
  urgency VARCHAR(20), -- 'critical', 'high', 'medium', 'low'
  analyzed_at TIMESTAMP
);

-- Aggregated metrics
CREATE TABLE sentiment_metrics (
  id UUID PRIMARY KEY,
  date DATE,
  source VARCHAR(50),
  category VARCHAR(100),
  total_feedback INTEGER,
  positive_count INTEGER,
  negative_count INTEGER,
  neutral_count INTEGER,
  avg_sentiment DECIMAL(3,2),
  top_topics JSONB
);
```

#### Step 2: Build Sentiment Analysis Function

```javascript
// Your tasks:
// 1. Create prompt for sentiment analysis
// 2. Request structured output
// 3. Parse and store results
```

**Analysis Prompt:**

```javascript
const SENTIMENT_PROMPT = `
Analyze the sentiment and key information from this customer feedback.

Feedback: "${feedbackText}"

Provide your analysis in JSON format:
{
  "sentiment": "positive|negative|neutral|mixed",
  "sentiment_score": <number from -1.0 (very negative) to 1.0 (very positive)>,
  "confidence": <number from 0.0 to 1.0>,
  "emotions": {
    "joy": <0.0-1.0>,
    "anger": <0.0-1.0>,
    "sadness": <0.0-1.0>,
    "fear": <0.0-1.0>,
    "surprise": <0.0-1.0>
  },
  "topics": ["topic1", "topic2"],
  "urgency": "critical|high|medium|low",
  "key_issues": ["issue1", "issue2"],
  "summary": "One sentence summary"
}

Guidelines:
- "critical" urgency = customer threatening to leave, extremely angry, service outage
- "high" urgency = significant problem, multiple complaints, dissatisfied
- Topics = what the feedback is about (e.g., "pricing", "shipping", "quality")
- Key issues = specific problems mentioned
`;
```

**Analysis Function:**

```javascript
async function analyzeSentiment(feedback) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini", // Cheaper model is fine for this
    messages: [
      {
        role: "system",
        content: "You are a sentiment analysis expert. Return only valid JSON.",
      },
      {
        role: "user",
        content: SENTIMENT_PROMPT.replace("${feedbackText}", feedback.text),
      },
    ],
    response_format: { type: "json_object" },
  });

  const analysis = JSON.parse(response.choices[0].message.content);

  // Store in database
  await db.sentimentAnalysis.create({
    feedback_id: feedback.id,
    ...analysis,
    analyzed_at: new Date(),
  });

  return analysis;
}
```

#### Step 3: Implement Batch Processing

```javascript
// Your tasks:
// 1. Create job queue for processing feedback
// 2. Process in batches to manage costs
// 3. Add retry logic for failures
// 4. Track processing progress
```

**Batch Processor:**

```javascript
const Queue = require("bull");
const sentimentQueue = new Queue("sentiment-analysis");

// Add feedback to queue
async function queueFeedback(feedback) {
  await sentimentQueue.add(
    "analyze",
    {
      feedbackId: feedback.id,
    },
    {
      attempts: 3,
      backoff: { type: "exponential", delay: 2000 },
    },
  );
}

// Process jobs
sentimentQueue.process("analyze", 5, async (job) => {
  const { feedbackId } = job.data;

  const feedback = await db.feedback.findById(feedbackId);
  const analysis = await analyzeSentiment(feedback);

  await db.feedback.update(feedbackId, {
    processed_at: new Date(),
  });

  // Check if urgent - send alert
  if (analysis.urgency === "critical") {
    await sendUrgentAlert(feedback, analysis);
  }

  return analysis;
});

// Batch process unprocessed feedback
async function processBacklog() {
  const unprocessed = await db.feedback.findMany({
    processed_at: null,
    limit: 100,
  });

  console.log(`Processing ${unprocessed.length} feedback items...`);

  for (const feedback of unprocessed) {
    await queueFeedback(feedback);
  }
}
```

#### Step 4: Build Aggregation Pipeline

```javascript
// Your tasks:
// 1. Aggregate daily metrics
// 2. Calculate trends
// 3. Extract top topics per category
// 4. Identify sentiment changes
```

**Daily Aggregation:**

```javascript
async function aggregateDailyMetrics(date) {
  const results = await db.query(
    `
    SELECT 
      DATE(created_at) as date,
      source,
      metadata->>'category' as category,
      COUNT(*) as total_feedback,
      SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
      SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
      SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
      AVG(sentiment_score) as avg_sentiment,
      jsonb_agg(DISTINCT topic) FILTER (WHERE topic IS NOT NULL) as all_topics
    FROM feedback f
    JOIN sentiment_analysis sa ON f.id = sa.feedback_id,
    jsonb_array_elements_text(sa.topics) as topic
    WHERE DATE(f.created_at) = $1
    GROUP BY DATE(created_at), source, metadata->>'category'
  `,
    [date],
  );

  // Store aggregated metrics
  for (const row of results) {
    await db.sentimentMetrics.create({
      date: row.date,
      source: row.source,
      category: row.category,
      total_feedback: row.total_feedback,
      positive_count: row.positive_count,
      negative_count: row.negative_count,
      neutral_count: row.neutral_count,
      avg_sentiment: row.avg_sentiment,
      top_topics: getTopTopics(row.all_topics, 5),
    });
  }
}

// Run daily (via cron job)
cron.schedule("0 1 * * *", () => {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  aggregateDailyMetrics(yesterday);
});
```

**Trend Detection:**

```javascript
async function detectTrends(category, days = 7) {
  const metrics = await db.sentimentMetrics.findMany({
    where: {
      category,
      date: { gte: daysAgo(days) },
    },
    orderBy: { date: "asc" },
  });

  // Calculate trend
  const scores = metrics.map((m) => m.avg_sentiment);
  const trend = linearRegression(scores);

  // Detect significant changes
  const recentAvg = average(scores.slice(-3));
  const previousAvg = average(scores.slice(0, -3));
  const change = ((recentAvg - previousAvg) / Math.abs(previousAvg)) * 100;

  return {
    trend: trend > 0 ? "improving" : "declining",
    change_percent: change.toFixed(1),
    is_significant: Math.abs(change) > 10,
  };
}
```

#### Step 5: Create Visualization Dashboard

```javascript
// Your tasks:
// 1. Build API endpoints for dashboard data
// 2. Create React components for charts
// 3. Implement time range filters
// 4. Add drill-down capabilities
```

**API Endpoints:**

```javascript
// Get sentiment overview
app.get("/api/sentiment/overview", async (req, res) => {
  const { startDate, endDate, category } = req.query;

  const data = await db.sentimentMetrics.aggregate({
    where: {
      date: { gte: startDate, lte: endDate },
      category: category || undefined,
    },
  });

  res.json({
    total_feedback: data.sum.total_feedback,
    avg_sentiment: data.avg.avg_sentiment,
    sentiment_distribution: {
      positive: data.sum.positive_count,
      negative: data.sum.negative_count,
      neutral: data.sum.neutral_count,
    },
    trend: await detectTrends(category, 30),
  });
});

// Get sentiment over time
app.get("/api/sentiment/timeline", async (req, res) => {
  const { startDate, endDate, category, granularity } = req.query;

  const metrics = await db.sentimentMetrics.findMany({
    where: {
      date: { gte: startDate, lte: endDate },
      category,
    },
    orderBy: { date: "asc" },
  });

  res.json(metrics);
});

// Get top topics
app.get("/api/sentiment/topics", async (req, res) => {
  const { startDate, endDate, limit = 10 } = req.query;

  const topics = await db.query(
    `
    SELECT 
      topic,
      COUNT(*) as mentions,
      AVG(sentiment_score) as avg_sentiment,
      SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_mentions
    FROM sentiment_analysis,
    jsonb_array_elements_text(topics) as topic
    WHERE analyzed_at >= $1 AND analyzed_at <= $2
    GROUP BY topic
    ORDER BY mentions DESC
    LIMIT $3
  `,
    [startDate, endDate, limit],
  );

  res.json(topics);
});
```

**Dashboard Components:**

```jsx
// Sentiment trend chart
function SentimentTrendChart({ data }) {
  return (
    <LineChart width={800} height={400} data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="date" />
      <YAxis domain={[-1, 1]} />
      <Tooltip />
      <Legend />
      <Line
        type="monotone"
        dataKey="avg_sentiment"
        stroke="#8884d8"
        name="Average Sentiment"
      />
    </LineChart>
  );
}

// Sentiment distribution pie chart
function SentimentDistribution({ positive, negative, neutral }) {
  const data = [
    { name: "Positive", value: positive, color: "#10b981" },
    { name: "Negative", value: negative, color: "#ef4444" },
    { name: "Neutral", value: neutral, color: "#6b7280" },
  ];

  return (
    <PieChart width={400} height={400}>
      <Pie
        data={data}
        dataKey="value"
        nameKey="name"
        cx="50%"
        cy="50%"
        fill="#8884d8"
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={entry.color} />
        ))}
      </Pie>
      <Tooltip />
      <Legend />
    </PieChart>
  );
}

// Top topics word cloud or list
function TopTopics({ topics }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {topics.map((topic) => (
        <div key={topic.topic} className="border rounded p-4">
          <h4 className="font-bold">{topic.topic}</h4>
          <p>{topic.mentions} mentions</p>
          <p
            className={
              topic.avg_sentiment > 0 ? "text-green-600" : "text-red-600"
            }
          >
            Avg sentiment: {topic.avg_sentiment.toFixed(2)}
          </p>
        </div>
      ))}
    </div>
  );
}
```

#### Step 6: Add Alerting and Insights

```javascript
// Your tasks:
// 1. Monitor for sentiment spikes
// 2. Alert on critical feedback
// 3. Generate weekly insights report
```

**Alert System:**

```javascript
async function checkAlertConditions() {
  // Check for negative sentiment spike
  const recentNegative = await db.query(`
    SELECT COUNT(*) as count
    FROM sentiment_analysis
    WHERE sentiment = 'negative'
    AND analyzed_at > NOW() - INTERVAL '1 hour'
  `);

  if (recentNegative[0].count > 10) {
    await sendAlert({
      type: "sentiment_spike",
      severity: "high",
      message: `${recentNegative[0].count} negative feedback items in last hour`,
      action_url: "/dashboard/recent-negative",
    });
  }

  // Check for critical feedback
  const critical = await db.sentimentAnalysis.findMany({
    where: {
      urgency: "critical",
      alerted: false,
    },
  });

  for (const item of critical) {
    await sendAlert({
      type: "critical_feedback",
      severity: "critical",
      feedback_id: item.feedback_id,
      message: item.summary,
      action_url: `/feedback/${item.feedback_id}`,
    });

    await db.sentimentAnalysis.update(item.id, { alerted: true });
  }
}

// Run every 5 minutes
setInterval(checkAlertConditions, 5 * 60 * 1000);
```

**Weekly Insights Report:**

```javascript
async function generateWeeklyReport() {
  const lastWeek = daysAgo(7);

  // Get metrics
  const metrics = await db.sentimentMetrics.findMany({
    where: { date: { gte: lastWeek } },
  });

  const totalFeedback = metrics.reduce((sum, m) => sum + m.total_feedback, 0);
  const avgSentiment = average(metrics.map((m) => m.avg_sentiment));
  const positivePercent =
    (metrics.reduce((sum, m) => sum + m.positive_count, 0) / totalFeedback) *
    100;

  // Get trends
  const trends = await detectTrends("all", 14);

  // Get top issues
  const topNegativeTopics = await db.query(
    `
    SELECT topic, COUNT(*) as count
    FROM sentiment_analysis,
    jsonb_array_elements_text(topics) as topic
    WHERE sentiment = 'negative'
    AND analyzed_at >= $1
    GROUP BY topic
    ORDER BY count DESC
    LIMIT 5
  `,
    [lastWeek],
  );

  const report = {
    period: "Last 7 days",
    summary: {
      total_feedback: totalFeedback,
      avg_sentiment: avgSentiment.toFixed(2),
      positive_percent: positivePercent.toFixed(1),
      trend: trends.trend,
      change: trends.change_percent,
    },
    top_issues: topNegativeTopics,
    recommendations: generateRecommendations(topNegativeTopics, trends),
  };

  // Send report via email or Slack
  await sendReport(report);

  return report;
}

function generateRecommendations(issues, trends) {
  const recommendations = [];

  if (trends.trend === "declining" && trends.is_significant) {
    recommendations.push(
      "⚠️ Sentiment declining significantly - investigate root causes",
    );
  }

  for (const issue of issues.slice(0, 3)) {
    recommendations.push(
      `🔍 Focus on "${issue.topic}" - ${issue.count} negative mentions`,
    );
  }

  return recommendations;
}
```

### Deliverables

- [ ] Sentiment analysis function with structured output
- [ ] Batch processing system with queue
- [ ] Database schema with metrics tables
- [ ] Daily aggregation pipeline
- [ ] Interactive dashboard with 3+ chart types
- [ ] Alerting system for critical feedback
- [ ] Weekly insights report generator

### Testing Checklist

- [ ] Test with various sentiment types (positive, negative, mixed)
- [ ] Verify batch processing handles 100+ items
- [ ] Check aggregation accuracy (spot-check calculations)
- [ ] Test dashboard with different date ranges
- [ ] Verify alerts trigger correctly
- [ ] Test with real feedback data (or realistic fake data)
- [ ] Check performance with 10,000+ feedback items

### Bonus Challenges

- Compare sentiment across competitors (if data available)
- Implement aspect-based sentiment (sentiment per feature)
- Add automated response suggestions for negative feedback
- Create sentiment prediction (predict upcoming trends)
- Multi-language support
- Real-time dashboard (WebSocket updates)
- Export reports as PDF
- A/B test different products/features based on sentiment

### Real-World Applications

- **E-commerce**: Monitor product reviews, identify issues early
- **SaaS**: Track customer satisfaction, predict churn
- **Support**: Prioritize urgent tickets, improve response time
- **Product**: Identify feature requests, validate ideas
- **Marketing**: Monitor campaign reception, brand sentiment
- **HR**: Analyze employee feedback, improve culture

### Performance Tips

```javascript
// Use cheaper models for bulk analysis
const model = batchSize > 50 ? 'gpt-4o-mini' : 'gpt-4o';

// Cache common analyses
const cacheKey = `sentiment:${hash(text)}`;
const cached = await cache.get(cacheKey);

// Process in parallel (but rate limit)
const chunks = chunkArray(feedback, 10);
for (const chunk of chunks) {
  await Promise.all(chunk.map(analyzeSentiment));
  await sleep(1000); // Respect rate limits
}

// Use database indexes
CREATE INDEX idx_sentiment_date ON sentiment_analysis(analyzed_at);
CREATE INDEX idx_feedback_processed ON feedback(processed_at);
```

---

# Phase Completion Checklist

## Phase 1: Chatbot Foundation ✅

- [ ] RAG system with vector database
- [ ] Conversation memory management
- [ ] Function calling implementation
- [ ] All three projects tested and working

## Phase 2: AI Application Types ✅

- [ ] Image generation OR manipulation tool
- [ ] Document processing system
- [ ] Code review assistant
- [ ] Sentiment analysis dashboard
- [ ] At least 3 of 4 projects completed

---

# Next Steps After Completion

## Phase 3: Production Readiness (Future)

Once you've completed Phases 1 and 2, you'll be ready for:

- **Evaluation Systems**: Test AI output quality systematically
- **Cost Monitoring**: Track and optimize API spending
- **Error Handling**: Build robust retry and fallback logic
- **Observability**: Log and trace AI operations
- **A/B Testing**: Compare prompts and models
- **Fine-tuning**: Train custom models for specific tasks

## Skills You'll Have Gained

**Technical:**

- AI API integration (OpenAI, Anthropic, etc.)
- Vector databases and embeddings
- Prompt engineering across domains
- Structured output extraction
- Batch and real-time processing
- Data pipeline design

**AI-Specific:**

- When to use different model sizes
- Cost vs quality tradeoffs
- Managing context windows
- Handling AI failures gracefully
- Evaluating AI output quality
- Prompt debugging techniques

**Software Engineering:**

- Queue-based architectures
- Webhook integrations
- Data aggregation pipelines
- Dashboard design
- API design for AI features

---

# Resources and Learning Materials

## Essential Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

## Tools to Explore

- **Vector Databases**: Pinecone, Weaviate, ChromaDB, Qdrant
- **AI Frameworks**: LangChain, LlamaIndex, Haystack
- **Observability**: LangSmith, Helicone, Weights & Biases
- **Development**: Cursor IDE, GitHub Copilot

## Communities

- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) - Open source AI
- [LangChain Discord](https://discord.gg/langchain)
- [OpenAI Developer Forum](https://community.openai.com/)

---

# Tips for Success

## 1. Start Simple

Don't try to build everything at once. Get the basic version working first, then add features.

## 2. Test with Real Data

Use actual documents, code, or feedback from your company. Real data reveals edge cases.

## 3. Iterate on Prompts

Your first prompt won't be perfect. Save different versions and compare results.

## 4. Monitor Costs

Set up alerts when spending exceeds thresholds. AI APIs can get expensive quickly.

## 5. Handle Failures Gracefully

AI will sometimes give bad outputs. Always validate and have fallbacks.

## 6. Document Your Learnings

Keep notes on what prompts worked well, what parameters gave best results, etc.

## 7. Ask for Feedback

Show your projects to team members and get their input on usefulness and accuracy.

---

# Weekly Check-ins with Mentor

Schedule regular check-ins to:

- Demo what you've built
- Discuss challenges and blockers
- Get code review feedback
- Plan next steps
- Explore real use cases at the company

**Suggested Format:**

- 10 min: Demo new functionality
- 10 min: Discuss technical challenges
- 10 min: Review code/architecture
- 10 min: Plan next week's work

---

# Questions to Ask Yourself

After each project:

- ✅ Does this work reliably with real data?
- ✅ How accurate are the AI outputs?
- ✅ What's the cost per operation?
- ✅ How long does it take to process?
- ✅ What edge cases break it?
- ✅ Would this be useful in production?
- ✅ What would I do differently next time?

---

# Final Notes

**Remember:** The goal isn't perfection - it's learning. Some things will break, some prompts won't work, and some approaches will need refactoring. That's all part of the process.

**Ship early:** Get a basic version working and get feedback. Don't wait for it to be perfect.

**Have fun:** AI development is creative work. Experiment, try new things, and enjoy the process of building intelligent systems!

Good luck! 🚀

---

**Document Version:** 1.0  
**Last Updated:** January 2026
