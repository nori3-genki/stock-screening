require('dotenv').config(); 
console.log('API_KEY:', process.env.ALPHA_VANTAGE_API_KEY ? 'Loaded' : 'Not found'); 
