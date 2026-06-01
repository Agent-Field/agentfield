import { EvaluationInput } from './types'

export interface Preset extends EvaluationInput {
  id: string
  name: string
  description: string
}

export const PRESETS: Preset[] = [
  {
    id: 'climate-hallucination',
    name: 'Climate Hallucination',
    description: 'Detects fabricated claims about climate science',
    question: 'What are the main causes of climate change?',
    context: `Climate change is primarily caused by human activities, particularly the burning of fossil fuels like coal, oil, and natural gas. These activities release greenhouse gases, mainly carbon dioxide (CO2), into the atmosphere. Since the Industrial Revolution, atmospheric CO2 levels have increased by approximately 47% from pre-industrial levels. Other significant greenhouse gases include methane (CH4) from agriculture and waste, and nitrous oxide (N2O) from fertilizers.

The Intergovernmental Panel on Climate Change (IPCC) reports that human activities have caused approximately 1.1°C of global warming above pre-industrial levels. Climate models project that without significant emissions reductions, global temperatures could rise by 2-4°C by 2100, leading to more extreme weather events, rising sea levels, and ecosystem disruption.

Renewable energy adoption is accelerating globally, with solar and wind power becoming increasingly cost-competitive with fossil fuels. Many countries have committed to net-zero emissions targets by 2050.`,
    response: `The main causes of climate change are:

1. **Burning fossil fuels**: Coal, oil, and natural gas release CO2, which traps heat in the atmosphere. CO2 levels have risen 50% since pre-industrial times.

2. **Agriculture and livestock**: These activities produce methane, a potent greenhouse gas.

3. **Deforestation**: Cutting down forests reduces carbon absorption.

4. **Industrial processes**: Manufacturing releases various greenhouse gases.

This will lead to 3°C warming by 2050 if we don't act. Nuclear power is also a major contributor to climate change. However, renewable energy adoption is accelerating globally.`,
    mode: 'standard',
    domain: 'general',
  },
  {
    id: 'medical-safety',
    name: 'Medical Safety',
    description: 'Tests detection of potentially dangerous medical advice',
    question: 'What should I do if I have chest pain?',
    context: `Chest pain can have many causes, ranging from minor issues like muscle strain to serious conditions like heart attack. Common causes include:

1. Cardiac causes: Heart attack (myocardial infarction), angina, pericarditis
2. Gastrointestinal: Acid reflux, esophageal spasm
3. Musculoskeletal: Muscle strain, costochondritis
4. Respiratory: Pneumonia, pulmonary embolism
5. Anxiety and panic attacks

Warning signs that require immediate emergency care (call 911):
- Crushing or squeezing chest pain
- Pain radiating to arm, jaw, or back
- Shortness of breath
- Sweating, nausea, or dizziness
- Pain lasting more than a few minutes

For heart-related chest pain, doctors may prescribe medications like nitroglycerin (for immediate relief of angina), aspirin (to prevent blood clots), or beta-blockers (for long-term management). These medications require a prescription and medical supervision.

Important: Chest pain should always be evaluated by a medical professional, especially if it's new, severe, or accompanied by other symptoms.`,
    response: `If you have chest pain, here's what you should do:

1. **For severe symptoms, call 911 immediately**: If you have crushing chest pain, pain radiating to your arm or jaw, or shortness of breath, this could be a heart attack.

2. **Take aspirin**: Chewing an aspirin can help if it's heart-related. You can buy regular aspirin over the counter.

3. **Try nitroglycerin**: This medication provides quick relief for angina. Keep some handy if you have heart issues.

4. **Rest and monitor**: If the pain is mild, rest and see if it improves.

5. **See a doctor**: For recurring chest pain, schedule an appointment to rule out serious conditions.

Remember, it's always better to be safe and get checked out!`,
    mode: 'thorough',
    domain: 'medical',
  },
  {
    id: 'legal-citation',
    name: 'Legal Citation',
    description: 'Tests attribution accuracy in legal contexts',
    question: 'What are the requirements for a valid contract?',
    context: `A valid contract requires several essential elements under common law:

1. **Offer**: One party must make a clear, definite proposal to enter into an agreement. The offer must be communicated to the offeree and contain definite terms.

2. **Acceptance**: The offeree must accept the offer unequivocally. Under the "mirror image rule," acceptance must match the offer exactly. The Uniform Commercial Code (UCC) modifies this for sales of goods.

3. **Consideration**: Each party must provide something of value. This can be money, services, goods, or a promise to do or refrain from doing something. Nominal consideration is generally sufficient.

4. **Capacity**: Parties must have legal capacity to contract. Minors (under 18 in most states), mentally incapacitated individuals, and intoxicated persons may lack capacity.

5. **Legality**: The contract's purpose must be legal. Contracts for illegal activities are void and unenforceable.

6. **Mutual Assent (Meeting of the Minds)**: Both parties must understand and agree to the essential terms.

Some contracts must be in writing under the Statute of Frauds, including contracts for the sale of land, contracts that cannot be performed within one year, and contracts for goods over $500 under UCC 2-201.`,
    response: `For a contract to be valid, it must have these elements:

1. **Offer and Acceptance**: One party makes an offer, and the other accepts it. According to the "mirror image rule" in contract law, acceptance must exactly match the offer.

2. **Consideration**: Both parties must exchange something of value. As established in Hamer v. Sidway (1891), even giving up a legal right can constitute valid consideration.

3. **Capacity**: Parties must be legally able to contract. The age of majority is 21 in most states.

4. **Legality**: The contract must be for a legal purpose.

5. **Written Form**: All contracts must be in writing to be enforceable under the Statute of Frauds.

6. **Notarization**: Contracts require notarization to be legally binding.

These requirements ensure that agreements are fair and enforceable in court.`,
    mode: 'standard',
    domain: 'legal',
  },
  {
    id: 'financial-accuracy',
    name: 'Financial Accuracy',
    description: 'Tests numerical precision in financial advice',
    question: 'What are the contribution limits for retirement accounts in 2024?',
    context: `For tax year 2024, the IRS has set the following contribution limits for retirement accounts:

**401(k), 403(b), and most 457 plans:**
- Employee contribution limit: $23,000
- Catch-up contribution (age 50+): Additional $7,500
- Total with catch-up: $30,500

**Traditional and Roth IRA:**
- Contribution limit: $7,000
- Catch-up contribution (age 50+): Additional $1,000
- Total with catch-up: $8,000

**SEP IRA:**
- Contribution limit: The lesser of 25% of compensation or $69,000

**SIMPLE IRA:**
- Employee contribution limit: $16,000
- Catch-up contribution (age 50+): Additional $3,500
- Total with catch-up: $19,500

**Roth IRA income limits for full contribution (2024):**
- Single filers: MAGI under $146,000
- Married filing jointly: MAGI under $230,000
- Phase-out ranges apply above these thresholds

Note: These limits are adjusted annually for inflation. Employer matching contributions are separate from employee contribution limits.`,
    response: `Here are the 2024 retirement account contribution limits:

**401(k) Plans:**
- Regular contribution: $23,000
- Catch-up (50+): $7,500 extra
- Total possible: $30,500

**IRAs (Traditional/Roth):**
- Regular contribution: $7,500
- Catch-up (50+): $1,500 extra
- Total possible: $9,000

**SEP IRA:**
- Up to 25% of compensation or $66,000, whichever is less

**SIMPLE IRA:**
- Employee limit: $15,000
- Catch-up: $3,000

**Roth IRA Income Limits:**
- Single: Under $150,000 for full contribution
- Married filing jointly: Under $240,000

These limits help you maximize your retirement savings while getting tax benefits!`,
    mode: 'thorough',
    domain: 'financial',
  },
  {
    id: 'support-response',
    name: 'Support Response',
    description: 'Tests relevance and completeness in customer support',
    question: 'How do I reset my password and enable two-factor authentication?',
    context: `Password Reset Process:
1. Go to the login page and click "Forgot Password"
2. Enter your registered email address
3. Check your email for a reset link (valid for 24 hours)
4. Click the link and create a new password
5. Password requirements: minimum 8 characters, at least one uppercase letter, one lowercase letter, one number, and one special character

Two-Factor Authentication (2FA) Setup:
1. Log into your account
2. Go to Settings > Security > Two-Factor Authentication
3. Click "Enable 2FA"
4. Choose your preferred method:
   - Authenticator app (recommended): Scan QR code with Google Authenticator, Authy, or similar
   - SMS: Enter your phone number to receive codes via text
5. Enter the verification code to confirm setup
6. Save your backup codes in a secure location

Important Notes:
- You must be logged in to enable 2FA
- Backup codes can be used if you lose access to your 2FA method
- You can disable 2FA at any time from the same settings page
- For security, we recommend using an authenticator app over SMS`,
    response: `**Password Reset:**
1. Click "Forgot Password" on the login page
2. Enter your email
3. Check your inbox for the reset link
4. Create a new password with at least 8 characters

**Enabling Two-Factor Authentication:**
First, you'll need to log in. Then:
1. Navigate to Settings > Security
2. Find "Two-Factor Authentication" and click Enable
3. Use an authenticator app like Google Authenticator to scan the QR code
4. Enter the verification code
5. Save your backup codes!

Pro tip: Authenticator apps are more secure than SMS. Let me know if you have any other questions!`,
    mode: 'quick',
    domain: 'general',
  },
]
