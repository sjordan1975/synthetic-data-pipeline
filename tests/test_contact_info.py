"""
Tests for ContactInfo model.

This module tests the ContactInfo Pydantic model including:
- Valid data creation
- Email validation and normalization
- Phone validation and cleaning
- Error handling for invalid data
"""

import pytest
from pydantic import ValidationError
from schemas.models import ContactInfo


class TestContactInfoValidData:
    """Test ContactInfo with valid data."""
    
    def test_valid_contact_info_creation(self, valid_contact_info):
        """Test that valid contact info creates successfully."""
        assert valid_contact_info.name == "John Doe"
        assert valid_contact_info.location == "San Francisco, CA"
    
    def test_email_normalization(self):
        """Test that email addresses are normalized to lowercase."""
        contact = ContactInfo(
            name="John Doe",
            email="JOHN.DOE@EXAMPLE.COM",  # Uppercase
            phone_region="US",
            phone="(202) 555-0173",
            location="San Francisco, CA"
        )
        assert contact.email == "john.doe@example.com"  # Should be lowercase
    
    def test_phone_cleaning_various_formats(self):
        """Test that phone numbers are normalized to E164 format."""
        test_cases = [
            ("(202) 555-0173", "+12025550173"),  # US format
            ("202.555.0173", "+12025550173"),  # US format
            ("202 555 0173", "+12025550173"),  # US format
            ("202-555-0173", "+12025550173"),  # US format
            ("+44 20 7946 0958", "+442079460958"),  # UK format
            ("+49 30 12345678", "+493012345678"),   # Germany format
        ]
        
        for input_phone, expected_e164 in test_cases:
            contact = ContactInfo(
                name="Test User",
                email="test@example.com",
                phone_region="US",
                phone=input_phone,
                location="Test City"
            )
            assert contact.phone == expected_e164
    
    def test_minimum_valid_phone(self):
        """Test that exactly 10 digits work for phone numbers."""
        contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone_region="US",
            phone="2025550173",  # Exactly 10 digits
            location="Test City"
        )
        assert contact.phone.startswith('+')  # Should be in E164 format
    
    def test_long_phone_numbers(self):
        """Test that phone numbers longer than 10 digits work."""
        contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone="+1 (202) 555-0173",  # 11 digits with country code
            location="Test City"
        )
        assert contact.phone.startswith('+')  # Should be in E164 format

    def test_local_phone_requires_region(self):
        """Test that non-E.164 numbers require phone_region."""
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="Test User",
                email="test@example.com",
                phone="2025550173",
                location="Test City"
            )

        assert "phone_region is required" in str(exc_info.value)

    def test_local_phone_with_region_parses(self):
        """Test that local numbers parse when phone_region is provided."""
        contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone_region="US",
            phone="2025550173",
            location="Test City"
        )

        assert contact.phone == "+12025550173"

    def test_au_local_phone_with_region_parses(self):
        """Test AU local parsing with explicit region context."""
        contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone_region="AU",
            phone="0420333333",
            location="Sydney, AU"
        )

        assert contact.phone == "+61420333333"


class TestContactInfoEmailValidation:
    """Test email validation in ContactInfo."""
    
    @pytest.mark.parametrize("valid_email", [
        "simple@example.com",
        "very.common@example.com",
        "disposable.style.email.with+symbol@example.com",
        "other.email-with-dash@example.com",
        "fully-qualified-domain@example.com",
        "user.name+tag+sorting@example.com",
        "x@example.com",
        "example-indeed@strange-example.com",
        "example@s.example",
        "test.email+alex@leetcode.cn",
        "user@sub.domain.com"
    ])
    def test_valid_email_formats(self, valid_email):
        """Test that various valid email formats are accepted."""
        contact = ContactInfo(
            name="Test User",
            email=valid_email,
            phone_region="US",
            phone="2025550173",
            location="Test City"
        )
        assert "@" in contact.email
    
    def test_invalid_email_rejection(self, sample_invalid_emails):
        """Test that invalid emails are rejected."""
        # Test specific invalid emails that should definitely fail
        definitely_invalid = [
            "invalid-email",  # No @ symbol
            "user@",  # Missing domain
            "@domain.com",  # Missing local part
            "user@domain",  # Missing TLD
        ]
        
        for invalid_email in definitely_invalid:
            with pytest.raises(ValidationError) as exc_info:
                ContactInfo(
                    name="Test User",
                    email=invalid_email,
                    phone_region="US",
                    phone="2025550173",
                    location="Test City"
                )
            # EmailStr has different error messages, just check that validation fails
            assert "email" in str(exc_info.value).lower()
    
    def test_specific_invalid_email_cases(self):
        """Test specific invalid email cases with detailed error checking."""
        invalid_cases = [
            ("plainaddress", "Missing @ symbol"),
            ("@missing-local.org", "Missing local part"),
            ("username@", "Missing domain"),
            ("username@.com", "Domain starts with dot"),
            ("username@.com.com", "Domain starts with dot"),
            (".username@yahoo.com", "Local part starts with dot"),
            ("username@yahoo.com.", "Domain ends with dot"),
            ("username@yahoo..com", "Double dot in domain"),
            ("user..name@domain.com", "Double dots in local part")
        ]
        
        for invalid_email, description in invalid_cases:
            with pytest.raises(ValidationError) as exc_info:
                ContactInfo(
                    name="Test User",
                    email=invalid_email,
                    phone_region="US",
                    phone="2025550173",
                    location="Test City"
                )
            # EmailStr has different error messages, just check that validation fails
            assert "email" in str(exc_info.value).lower()
    
    def test_edge_case_emails(self):
        """Test edge case emails with EmailStr validation."""
        # EmailStr is more strict, so these should fail
        edge_cases = [
            "admin@mailserver1",  # Internal email without TLD - EmailStr rejects
            "username@com",  # Single word domain - EmailStr rejects
        ]
        
        for email in edge_cases:
            with pytest.raises(ValidationError):
                ContactInfo(
                    name="Test User",
                    email=email,
                    phone_region="US",
                    phone="2025550173",
                    location="Test City"
                )


class TestContactInfoPhoneValidation:
    """Test phone validation in ContactInfo."""
    
    def test_invalid_phone_rejection(self, sample_invalid_phones):
        """Test that invalid phone numbers are rejected."""
        for invalid_phone in sample_invalid_phones:
            with pytest.raises(ValidationError) as exc_info:
                ContactInfo(
                    name="Test User",
                    email="test@example.com",
                    phone_region="US",
                    phone=invalid_phone,
                    location="Test City"
                )
            assert "phone" in str(exc_info.value).lower()
    
    def test_phone_with_letters_rejected(self):
        """Test that phone numbers with letters are rejected."""
        invalid_phones = [
            "202-abc-defg",
            "phone-number",
            "123-456-789a",
            "202-555- PHONE"
        ]
        
        for invalid_phone in invalid_phones:
            with pytest.raises(ValidationError) as exc_info:
                ContactInfo(
                    name="Test User",
                    email="test@example.com",
                    phone_region="US",
                    phone=invalid_phone,
                    location="Test City"
                )
            assert "phone" in str(exc_info.value).lower()
    
    def test_phone_boundary_conditions(self):
        """Test phone number boundary conditions."""
        # Test exactly 9 digits (should fail)
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="Test User",
                email="test@example.com",
                phone_region="US",
                phone="555123456",  # Only 9 digits
                location="Test City"
            )
        assert "phone" in str(exc_info.value).lower()
        
        # Test exactly 10 digits (should pass)
        contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone_region="US",
            phone="2025550173",  # Exactly 10 digits
            location="Test City"
        )
        assert contact.phone.startswith('+')  # Should be in E164 format


class TestContactInfoOtherFields:
    """Test other fields in ContactInfo."""
    
    def test_name_validation(self):
        """Test name field validation."""
        # Valid name
        contact = ContactInfo(
            name="John Doe",
            email="john@example.com",
            phone_region="US",
            phone="2025550173",
            location="San Francisco, CA"
        )
        assert contact.name == "John Doe"
        
        # Name too short (should fail)
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="J",  # Too short (min_length=2)
                email="john@example.com",
                phone_region="US",
                phone="2025550173",
                location="San Francisco, CA"
            )
        assert "String should have at least 2 characters" in str(exc_info.value)
    
    def test_location_validation(self):
        """Test location field validation."""
        # Valid location
        contact = ContactInfo(
            name="John Doe",
            email="john@example.com",
            phone_region="US",
            phone="2025550173",
            location="SFO"  # 3 characters, valid
        )
        assert contact.location == "SFO"
        
        # Location too short (should fail)
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="John Doe",
                email="john@example.com",
                phone_region="US",
                phone="2025550173",
                location="S"  # Too short (min_length=3)
            )
        assert "String should have at least 3 characters" in str(exc_info.value)


class TestContactInfoEdgeCases:
    """Test edge cases and unusual but valid inputs."""
    
    def test_email_with_unicode_characters(self):
        """Test email with unicode characters in name and location."""
        contact = ContactInfo(
            name="José García",
            email="jose.garcia@example.com",  # ASCII version
            phone_region="US",
            phone="2025550173",
            location="São Paulo"
        )
        assert "jose.garcia@example.com" == contact.email
        assert "José García" == contact.name
        assert "São Paulo" == contact.location
    
    def test_phone_with_many_digits(self):
        """Test phone number with many digits (international format)."""
        contact = ContactInfo(
            name="International User",
            email="user@example.com",
            phone="+44 20 7946 0958",  # UK phone number
            location="London, UK"
        )
        assert len(contact.phone) > 10  # Should have more than 10 digits
    
    def test_name_with_special_characters(self):
        """Test name with special characters."""
        contact = ContactInfo(
            name="John O'Connor-Smith Jr.",
            email="john@example.com",
            phone_region="US",
            phone="2025550173",
            location="New York, NY"
        )
        assert "John O'Connor-Smith Jr." == contact.name
    
    def test_empty_string_handling(self):
        """Test that empty strings are handled appropriately."""
        # Empty name should fail (required field)
        with pytest.raises(ValidationError):
            ContactInfo(
                name="",  # Empty string
                email="test@example.com",
                phone_region="US",
                phone="2025550173",
                location="Test City"
            )
        
        # Empty email should fail (required field)
        with pytest.raises(ValidationError):
            ContactInfo(
                name="Test User",
                email="",  # Empty string
                phone_region="US",
                phone="2025550173",
                location="Test City"
            )


class TestContactInfoErrorMessages:
    """Test that error messages are clear and helpful."""
    
    def test_email_error_message_clarity(self):
        """Test that email error messages are clear."""
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="Test User",
                email="not-an-email",
                phone_region="US",
                phone="2025550173",
                location="Test City"
            )
        
        errors = exc_info.value.errors()
        email_error = next(e for e in errors if e['loc'] == ('email',))
        # EmailStr provides detailed error messages
        assert len(email_error['msg']) > 0  # Just check there's an error message
    
    def test_phone_error_message_clarity(self):
        """Test that phone error messages are clear."""
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="Test User",
                email="test@example.com",
                phone_region="US",
                phone="123",
                location="Test City"
            )
        
        errors = exc_info.value.errors()
        phone_error = next(e for e in errors if e['loc'] == ('phone',))
        assert "Invalid phone number format" in phone_error['msg']
    
    def test_multiple_field_errors(self):
        """Test that multiple validation errors are reported."""
        with pytest.raises(ValidationError) as exc_info:
            ContactInfo(
                name="J",  # Too short
                email="invalid-email",  # Invalid format
                phone_region="US",
                phone="123",  # Too short
                location="S"  # Too short
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 4  # Should have 4 validation errors
        
        # Check that all expected fields have errors
        error_locs = {error['loc'][0] for error in errors}
        assert error_locs == {'name', 'email', 'phone', 'location'}
